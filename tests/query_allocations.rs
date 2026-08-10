//! Allocation accounting for variable-path reads.
//!
//! Reading a single field out of a variable that holds a large object must cost
//! a small constant number of allocations, independent of how big the variable
//! is. The counting allocator below makes that measurable instead of inferred:
//! it records every allocation made on the measuring thread while a compiled
//! program runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;

use vrl::compiler::{Program, SecretTarget, TargetValueRef, TimeZone, runtime::Runtime};
use vrl::value::{KeyString, Secrets, Value};

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

fn record(size: usize) {
    let _ = ENABLED.try_with(|enabled| {
        if !enabled.get() {
            return;
        }
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        let _ = BYTES.try_with(|c| c.set(c.get() + size as u64));
    });
}

struct CountingAllocator;

// SAFETY: every method forwards to `System` with the arguments it was given.
// The extra work is thread-local counter bookkeeping, which allocates nothing.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug)]
struct Usage {
    allocs: u64,
    bytes: u64,
}

fn measure<T>(f: impl FnOnce() -> T) -> (Usage, T) {
    ALLOCS.with(|c| c.set(0));
    BYTES.with(|c| c.set(0));
    ENABLED.with(|c| c.set(true));
    let out = f();
    ENABLED.with(|c| c.set(false));
    let usage = Usage {
        allocs: ALLOCS.with(Cell::get),
        bytes: BYTES.with(Cell::get),
    };
    (usage, out)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Number of single-field reads per measured program.
const READS: usize = 100;

/// Object with `width` top-level keys, each holding a nested object of `width`
/// string values. Deep-cloning it costs thousands of allocations; cloning one
/// leaf out of it costs one.
fn wide_object(width: usize) -> Value {
    let mut outer = BTreeMap::new();
    for i in 0..width {
        let mut inner = BTreeMap::new();
        for j in 0..width {
            inner.insert(
                KeyString::from(format!("inner_key_{j}")),
                Value::from(format!(
                    "value {i}/{j} with enough text to own a heap buffer"
                )),
            );
        }
        outer.insert(KeyString::from(format!("k{i}")), Value::from(inner));
    }
    Value::from(outer)
}

fn event() -> Value {
    let mut root = BTreeMap::new();
    root.insert(KeyString::from("big"), wide_object(20));
    root.insert(KeyString::from("small"), wide_object(1));
    Value::from(root)
}

fn compile(source: &str) -> Program {
    vrl::compiler::compile(source, &vrl::stdlib::all())
        .unwrap_or_else(|diagnostics| panic!("failed to compile:\n{source}\n{diagnostics:?}"))
        .program
}

/// `field = <source field on the event>`, then `READS` copies of `read`.
fn program(field: &str, read: &str) -> Program {
    let mut source = format!("_v = .{field}\n");
    for _ in 0..READS {
        source.push_str(read);
        source.push('\n');
    }
    compile(&source)
}

fn run(program: &Program, target_value: &mut Value) {
    let mut metadata = Value::from(BTreeMap::new());
    let mut secrets = Secrets::new();
    let mut target = TargetValueRef {
        value: target_value,
        metadata: &mut metadata,
        secrets: &mut secrets,
    };
    target.insert_secret("unused", "unused");

    let mut runtime = Runtime::default();
    runtime
        .resolve(&mut target, program, &TimeZone::default())
        .expect("program must resolve");
}

/// Allocations attributable to the `READS` field reads alone, with the cost of
/// the setup assignment and the runtime scaffolding subtracted out.
fn per_read(field: &str, read: &str) -> Usage {
    let setup = program(field, "");
    let full = program(field, read);

    let value = event();
    // Warm caches and any lazily initialised statics outside the measurement.
    run(&setup, &mut value.clone());
    run(&full, &mut value.clone());

    let (setup_usage, ()) = measure(|| run(&setup, &mut value.clone()));
    let (full_usage, ()) = measure(|| run(&full, &mut value.clone()));

    assert!(
        full_usage.allocs >= setup_usage.allocs,
        "setup measured heavier than the full program: {setup_usage:?} vs {full_usage:?}"
    );

    Usage {
        allocs: (full_usage.allocs - setup_usage.allocs) / READS as u64,
        bytes: (full_usage.bytes - setup_usage.bytes) / READS as u64,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A single-field read must not scale with the size of the variable it reads
/// from. The bound is deliberately loose: the failure this guards against is a
/// deep clone of the whole variable, which costs hundreds of allocations, not a
/// one-or-two allocation drift in unrelated runtime bookkeeping.
const MAX_ALLOCS_PER_READ: u64 = 8;

#[test]
fn get_function_on_variable_does_not_clone_the_variable() {
    let read = r#"x = get!(_v, ["k1", "inner_key_1"])"#;
    let big = per_read("big", read);
    let small = per_read("small", read);

    println!("get!(_v, [..]) big object:   {big:?} per read");
    println!("get!(_v, [..]) small object: {small:?} per read");

    assert!(
        big.allocs <= MAX_ALLOCS_PER_READ,
        "reading one field via get() from a large object variable allocated \
         {big:?} per read; reading from a small one allocated {small:?}"
    );
}

#[test]
fn path_query_on_variable_does_not_clone_the_variable() {
    let read = "x = _v.k1.inner_key_1";
    let big = per_read("big", read);
    let small = per_read("small", read);

    println!("_v.k1.inner_key_1 big object:   {big:?} per read");
    println!("_v.k1.inner_key_1 small object: {small:?} per read");

    assert!(
        big.allocs <= MAX_ALLOCS_PER_READ,
        "reading one field via a path query from a large object variable \
         allocated {big:?} per read; reading from a small one allocated {small:?}"
    );
}
