//! VRL compile timing: synthetic else-if ladder and/or a real program file.
//!
//! ```text
//! cargo run --release --example else_if_ladder_compile -- \
//!   --arms 10,20,40,60 --fields 40
//!
//! cargo run --release --example else_if_ladder_compile -- \
//!   --program PATH --env-width 8 --env-depth 4 --warmup 2 --repeat 7
//! ```
//!
//! `--fields` is event-path assigns per synthetic ladder arm; `--locals` adds
//! local-variable assigns, and `--locals-shared` / `--fields-disjoint` control
//! whether those names repeat across arms (fixed live set) or are distinct
//! (live set grows with arm count).
//! `--env-width` / `--env-depth` / `--env-seed-fields` shape the starting event `Kind`
//! (incoming schema), independent of the program text.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use vrl::compiler::CompileConfig;
use vrl::compiler::state::ExternalEnv;
use vrl::value::Kind;
use vrl::value::kind::{Collection, Field};

// `find_enrichment_table_records` / `get_enrichment_table_record` are provided by
// vector's `enrichment` crate, not by `vrl::stdlib::all()`, so `--program` could not
// time real-world vector remap programs. Register compile-only stubs with the same
// arity and return kinds. They are fallible so `?? []` / `?? {}` stays well-typed. Only
// `compile`/`type_def` matter here; `resolve` is never executed by this harness.
mod enrichment_stubs {
    use vrl::compiler::prelude::*;

    const STUB_PARAMS: &[Parameter] = &[
        Parameter::required("table", kind::BYTES, "stub"),
        Parameter::required("condition", kind::OBJECT, "stub"),
        Parameter::optional("select", kind::ARRAY, "stub"),
        Parameter::optional("case_sensitive", kind::BOOLEAN, "stub"),
    ];

    #[derive(Clone, Copy, Debug)]
    pub struct FindEnrichmentTableRecords;

    impl Function for FindEnrichmentTableRecords {
        fn identifier(&self) -> &'static str {
            "find_enrichment_table_records"
        }
        fn usage(&self) -> &'static str {
            "stub"
        }
        fn category(&self) -> &'static str {
            "enrichment"
        }
        fn return_kind(&self) -> u16 {
            kind::ARRAY
        }
        fn examples(&self) -> &'static [Example] {
            &[]
        }
        fn parameters(&self) -> &'static [Parameter] {
            STUB_PARAMS
        }
        fn compile(
            &self,
            _state: &state::TypeState,
            _ctx: &mut FunctionCompileContext,
            _args: ArgumentList,
        ) -> Compiled {
            Ok(StubArrayFn.as_expr())
        }
    }

    #[derive(Debug, Clone)]
    struct StubArrayFn;

    impl FunctionExpression for StubArrayFn {
        fn resolve(&self, _: &mut Context) -> Resolved {
            Ok(Value::Array(vec![]))
        }
        fn type_def(&self, _: &state::TypeState) -> TypeDef {
            TypeDef::array(Collection::any()).fallible()
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct GetEnrichmentTableRecord;

    impl Function for GetEnrichmentTableRecord {
        fn identifier(&self) -> &'static str {
            "get_enrichment_table_record"
        }
        fn usage(&self) -> &'static str {
            "stub"
        }
        fn category(&self) -> &'static str {
            "enrichment"
        }
        fn return_kind(&self) -> u16 {
            kind::OBJECT
        }
        fn examples(&self) -> &'static [Example] {
            &[]
        }
        fn parameters(&self) -> &'static [Parameter] {
            STUB_PARAMS
        }
        fn compile(
            &self,
            _state: &state::TypeState,
            _ctx: &mut FunctionCompileContext,
            _args: ArgumentList,
        ) -> Compiled {
            Ok(StubObjectFn.as_expr())
        }
    }

    #[derive(Debug, Clone)]
    struct StubObjectFn;

    impl FunctionExpression for StubObjectFn {
        fn resolve(&self, _: &mut Context) -> Resolved {
            Ok(Value::Object(std::collections::BTreeMap::new()))
        }
        fn type_def(&self, _: &state::TypeState) -> TypeDef {
            TypeDef::object(Collection::any()).fallible()
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "Usage:
  else_if_ladder_compile [--arms N,N,...] [--fields F] [env opts] [--warmup W] [--repeat R]
  else_if_ladder_compile --program PATH [env opts] [--warmup W] [--repeat R]

  --arms             comma-separated else-if arm counts (default: 10,20,40,60,80)
  --fields           event-path assigns per synthetic ladder arm (default: 40)
  --locals L         local-variable assigns per synthetic ladder arm (default: 0)
  --coalesce C       fallible-call `??` literal-fallback assigns per arm (default: 0)
  --coalesce-chain N fallible calls chained by `??` per coalesce assign (default: 1)
  --locals-shared    reuse the same local names on every arm (fixed live set)
  --fields-disjoint  give every arm distinct event-field names (event Kind grows)
  --program          time compile of a VRL source file (skips synthetic ladder)
  --env-width W      known siblings per spine level (default: 0 = ExternalEnv::default())
  --env-depth D      spine nesting depth when width > 0 (default: 1)
  --env-seed-fields N  max typed known slots along spine+stubs (default: all)
  --warmup           discarded compiles before timing (default: 1)
  --repeat           timed compiles; report min/median/max ms (default: 3)

  Env Kind shape (width W, depth D): one spine path `.n.n.…` of length D; at each
  level, W-1 sibling stubs `.s0…`. Cost O(D*W). Types cycle a fixed palette by index.
"
    );
    process::exit(2);
}

fn parse_csv_usize(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid usize in list: {p:?}"))
        })
        .collect()
}

struct Args {
    arms: Vec<usize>,
    fields: usize,
    /// Local-variable assigns per arm.
    locals: usize,
    /// `to_string(.srcN) ?? "lit"` assigns per arm.
    coalesce: usize,
    /// Chained fallible calls per coalesce assign.
    coalesce_chain: usize,
    /// Reuse the same local names on every arm (fixed live set).
    locals_shared: bool,
    /// Give every arm distinct event-field names (wide event `Kind`).
    fields_disjoint: bool,
    warmup: usize,
    repeat: usize,
    program: Option<PathBuf>,
    env_width: usize,
    env_depth: usize,
    /// `None` = type every stub/leaf slot.
    env_seed_fields: Option<usize>,
}

fn parse_args() -> Args {
    let mut arms = vec![10, 20, 40, 60, 80];
    let mut fields = 40;
    let mut locals = 0;
    let mut coalesce = 0;
    let mut coalesce_chain = 1;
    let mut locals_shared = false;
    let mut fields_disjoint = false;
    let mut warmup = 1;
    let mut repeat = 3;
    let mut program = None;
    let mut env_width = 0;
    let mut env_depth = 1;
    let mut env_seed_fields = None;

    let mut argv = env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--arms" => {
                let v = argv.next().unwrap_or_else(|| usage());
                arms = parse_csv_usize(&v).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    usage();
                });
            }
            "--fields" => {
                let v = argv.next().unwrap_or_else(|| usage());
                fields = v.parse().unwrap_or_else(|_| usage());
            }
            "--locals" => {
                let v = argv.next().unwrap_or_else(|| usage());
                locals = v.parse().unwrap_or_else(|_| usage());
            }
            "--coalesce" => {
                let v = argv.next().unwrap_or_else(|| usage());
                coalesce = v.parse().unwrap_or_else(|_| usage());
            }
            "--coalesce-chain" => {
                let v = argv.next().unwrap_or_else(|| usage());
                coalesce_chain = v.parse().unwrap_or_else(|_| usage());
            }
            "--locals-shared" => locals_shared = true,
            "--fields-disjoint" => fields_disjoint = true,
            "--program" => {
                let v = argv.next().unwrap_or_else(|| usage());
                program = Some(PathBuf::from(v));
            }
            "--env-width" => {
                let v = argv.next().unwrap_or_else(|| usage());
                env_width = v.parse().unwrap_or_else(|_| usage());
            }
            "--env-depth" => {
                let v = argv.next().unwrap_or_else(|| usage());
                env_depth = v.parse().unwrap_or_else(|_| usage());
            }
            "--env-seed-fields" => {
                let v = argv.next().unwrap_or_else(|| usage());
                env_seed_fields = Some(v.parse().unwrap_or_else(|_| usage()));
            }
            "--warmup" => {
                let v = argv.next().unwrap_or_else(|| usage());
                warmup = v.parse().unwrap_or_else(|_| usage());
            }
            "--repeat" => {
                let v = argv.next().unwrap_or_else(|| usage());
                repeat = v.parse().unwrap_or_else(|_| usage());
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown arg: {other}");
                usage();
            }
        }
    }

    if program.is_none() && (arms.is_empty() || (fields == 0 && locals == 0 && coalesce == 0)) {
        eprintln!("--arms and one of --fields/--locals/--coalesce must be non-zero, or pass --program");
        usage();
    }
    if repeat == 0 {
        eprintln!("--repeat must be > 0");
        usage();
    }
    if env_width > 0 && env_depth == 0 {
        eprintln!("--env-depth must be > 0 when --env-width > 0");
        usage();
    }

    Args {
        arms,
        fields,
        locals,
        coalesce,
        coalesce_chain,
        locals_shared,
        fields_disjoint,
        warmup,
        repeat,
        program,
        env_width,
        env_depth,
        env_seed_fields,
    }
}

/// Deterministic leaf kinds for seeded env fields.
fn palette_kind(index: usize) -> Kind {
    match index % 6 {
        0 => Kind::bytes(),
        1 => Kind::integer(),
        2 => Kind::float(),
        3 => Kind::boolean(),
        4 => Kind::null(),
        _ => Kind::timestamp(),
    }
}

fn take_typed(counter: &mut usize, limit: usize) -> Option<Kind> {
    if *counter >= limit {
        return None;
    }
    let kind = palette_kind(*counter);
    *counter += 1;
    Some(kind)
}

/// Spine `.n.n.…` of length `depth`, with `width - 1` sibling stubs `.s0…` at each level.
/// Unknown fields remain `any`. Slot count is O(depth * width).
fn build_env_event_kind(width: usize, depth: usize, seed_limit: usize) -> Kind {
    fn level(width: usize, depth_left: usize, counter: &mut usize, limit: usize) -> Kind {
        let mut known: BTreeMap<Field, Kind> = BTreeMap::new();
        let stubs = width.saturating_sub(1);
        for s in 0..stubs {
            if let Some(kind) = take_typed(counter, limit) {
                known.insert(Field::from(format!("s{s}")), kind);
            }
        }
        if depth_left <= 1 {
            if let Some(kind) = take_typed(counter, limit) {
                known.insert(Field::from("n"), kind);
            }
        } else {
            known.insert(
                Field::from("n"),
                level(width, depth_left - 1, counter, limit),
            );
        }
        Kind::object(Collection::from_parts(known, Kind::any()))
    }

    level(width, depth, &mut 0, seed_limit)
}

fn build_external(args: &Args) -> ExternalEnv {
    if args.env_width == 0 {
        return ExternalEnv::default();
    }
    let seed_limit = args.env_seed_fields.unwrap_or(usize::MAX);
    let event = build_env_event_kind(args.env_width, args.env_depth, seed_limit);
    ExternalEnv::new_with_kind(event, Kind::object(Collection::any()))
}

fn env_label(args: &Args) -> String {
    if args.env_width == 0 {
        return "env=default".to_owned();
    }
    let seed = match args.env_seed_fields {
        None => "all".to_owned(),
        Some(n) => n.to_string(),
    };
    format!(
        "env-width={} env-depth={} env-seed-fields={seed}",
        args.env_width, args.env_depth
    )
}

/// Nested `if / else if` ladder with the same field set on every arm.
///
/// `locals_per_arm` adds local-variable assigns per arm. Names are disjoint per
/// arm (`l{i}_{j}`, so the live-local set grows with arms) unless `locals_shared`,
/// in which case every arm writes the same `l{j}` names (writes grow, live set
/// stays fixed). `fields_disjoint` does the same for event paths, which grows the
/// event `Kind` instead of the local env.
fn build_ladder(
    arm_count: usize,
    fields_per_arm: usize,
    locals_per_arm: usize,
    coalesce_per_arm: usize,
    coalesce_chain: usize,
    locals_shared: bool,
    fields_disjoint: bool,
) -> String {
    let coalesce_expr = |c: usize, fallback: &str| {
        let mut expr = String::new();
        for _ in 0..coalesce_chain.max(1) {
            expr.push_str(&format!("to_string(.src{c}) ?? "));
        }
        expr.push_str(&format!("\"{fallback}\""));
        expr
    };
    let field_name = |i: usize, f: usize| {
        if fields_disjoint {
            format!("f{i}_{f}")
        } else {
            format!("f{f}")
        }
    };
    let local_name = |i: usize, j: usize| {
        if locals_shared {
            format!("l{j}")
        } else {
            format!("l{i}_{j}")
        }
    };

    let mut out = String::with_capacity(arm_count * (80 + (fields_per_arm + locals_per_arm) * 40));
    out.push_str("eid = int!(.eid)\n");

    for i in 0..arm_count {
        if i == 0 {
            out.push_str(&format!("if eid == {i} {{\n"));
        } else {
            out.push_str(&format!("}} else if eid == {i} {{\n"));
        }
        for f in 0..fields_per_arm {
            out.push_str(&format!("  ._itl.{} = \"a{i}_f{f}\"\n", field_name(i, f)));
        }
        for c in 0..coalesce_per_arm {
            out.push_str(&format!(
                "  ._itl.c{c} = {}\n",
                coalesce_expr(c, &format!("a{i}_c{c}"))
            ));
        }
        for j in 0..locals_per_arm {
            out.push_str(&format!("  {} = \"a{i}_l{j}\"\n", local_name(i, j)));
        }
        out.push_str("  ._itl.class = \"NOTABLE\"\n");
        out.push_str(&format!("  ._itl.arm = {i}\n"));
    }
    out.push_str("} else {\n");
    for f in 0..fields_per_arm {
        out.push_str(&format!(
            "  ._itl.{} = \"else_f{f}\"\n",
            field_name(arm_count, f)
        ));
    }
    for c in 0..coalesce_per_arm {
        out.push_str(&format!(
            "  ._itl.c{c} = {}\n",
            coalesce_expr(c, &format!("else_c{c}"))
        ));
    }
    for j in 0..locals_per_arm {
        out.push_str(&format!("  {} = \"else_l{j}\"\n", local_name(arm_count, j)));
    }
    out.push_str("  ._itl.class = \"CONTEXT\"\n");
    out.push_str("  ._itl.arm = -1\n");
    out.push_str("}\n");
    out.push_str(".\n");
    out
}

fn compile_once(src: &str, fns: &[Box<dyn vrl::compiler::Function>], external: &ExternalEnv) {
    match vrl::compiler::compile_with_external(src, fns, external, CompileConfig::default()) {
        Ok(_) => {}
        Err(diags) => {
            eprintln!("compile failed:\n{diags:?}");
            process::exit(1);
        }
    }
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    }
}

fn time_src(
    label: &str,
    src: &str,
    warmup: usize,
    repeat: usize,
    fns: &[Box<dyn vrl::compiler::Function>],
    external: &ExternalEnv,
) {
    let src_bytes = src.len();
    let else_ifs = src.matches("else if").count();

    for _ in 0..warmup {
        compile_once(src, fns, external);
    }

    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        compile_once(src, fns, external);
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let med = median_ms(&mut samples);

    println!(
        "{label}  src_bytes={src_bytes}  else_if≈{else_ifs}  warmup={warmup}  repeat={repeat}"
    );
    println!("  min_ms={min:.1}  med_ms={med:.1}  max_ms={max:.1}");
}

fn main() {
    let args = parse_args();
    let mut fns = vrl::stdlib::all();
    fns.push(Box::new(enrichment_stubs::FindEnrichmentTableRecords));
    fns.push(Box::new(enrichment_stubs::GetEnrichmentTableRecord));
    let fns = fns;
    let external = build_external(&args);
    let env = env_label(&args);

    if let Some(path) = &args.program {
        let src = fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("read {}: {e}", path.display());
            process::exit(1);
        });
        time_src(
            &format!("program {}  {env}", path.display()),
            &src,
            args.warmup,
            args.repeat,
            &fns,
            &external,
        );
        return;
    }

    println!(
        "else-if ladder compile (fields/arm={}, locals/arm={}, coalesce/arm={} chain={} shared={}, {env}, warmup={}, repeat={})",
        args.fields,
        args.locals,
        args.coalesce,
        args.coalesce_chain,
        args.locals_shared,
        args.warmup,
        args.repeat
    );
    println!(
        "{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "arms", "src_bytes", "min_ms", "med_ms", "max_ms", "ms/arm^2"
    );

    for &n in &args.arms {
        let src = build_ladder(
            n,
            args.fields,
            args.locals,
            args.coalesce,
            args.coalesce_chain,
            args.locals_shared,
            args.fields_disjoint,
        );
        let src_bytes = src.len();

        for _ in 0..args.warmup {
            compile_once(&src, &fns, &external);
        }

        let mut samples = Vec::with_capacity(args.repeat);
        for _ in 0..args.repeat {
            let t0 = Instant::now();
            compile_once(&src, &fns, &external);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let med = median_ms(&mut samples);
        let per_n2 = if n > 0 {
            med / ((n as f64) * (n as f64))
        } else {
            0.0
        };

        println!("{n:>6}  {src_bytes:>10}  {min:>10.1}  {med:>10.1}  {max:>10.1}  {per_n2:>12.4}");
    }
}
