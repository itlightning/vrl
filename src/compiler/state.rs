use crate::path::PathPrefix;
use crate::value::{Kind, Value};
use std::collections::{HashMap, hash_map::Entry};
use std::ops::Deref;
use std::sync::Arc;

use super::{TypeDef, parser::ast::Ident, type_def::Details, value::Collection};

/// Shared local bindings: `TypeState` clones share until a write via [`Self::make_mut`].
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct SharedBindings(Arc<HashMap<Ident, Details>>);

impl SharedBindings {
    fn make_mut(&mut self) -> &mut HashMap<Ident, Details> {
        Arc::make_mut(&mut self.0)
    }

    fn into_map(self) -> HashMap<Ident, Details> {
        Arc::try_unwrap(self.0).unwrap_or_else(|arc| (*arc).clone())
    }

    /// Whether both handles point at the same allocation, i.e. neither side has written
    /// since the clone. Copy-on-write makes this a sound (one-sided) test for value equality.
    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Deref for SharedBindings {
    type Target = HashMap<Ident, Details>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub state: TypeState,
    pub result: TypeDef,
}

impl TypeInfo {
    #[must_use]
    pub fn new(state: impl Into<TypeState>, result: TypeDef) -> Self {
        Self {
            state: state.into(),
            result,
        }
    }

    #[must_use]
    pub fn map_result(self, f: impl FnOnce(TypeDef) -> TypeDef) -> Self {
        Self {
            state: self.state,
            result: f(self.result),
        }
    }
}

impl From<&TypeState> for TypeState {
    fn from(state: &TypeState) -> Self {
        state.clone()
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Default)]
pub struct TypeState {
    pub local: LocalEnv,
    pub external: ExternalEnv,
}

impl TypeState {
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            local: self.local.merge(other.local),
            external: self.external.merge(other.external),
        }
    }
}

/// Local environment, limited to a given scope.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct LocalEnv {
    pub(crate) bindings: SharedBindings,
}

impl LocalEnv {
    pub(crate) fn variable_idents(&self) -> impl Iterator<Item = &Ident> + '_ {
        self.bindings.keys()
    }

    pub(crate) fn variable(&self, ident: &Ident) -> Option<&Details> {
        self.bindings.get(ident)
    }

    pub(crate) fn variable_mut(&mut self, ident: &Ident) -> Option<&mut Details> {
        self.bindings.make_mut().get_mut(ident)
    }

    pub(crate) fn insert_variable(&mut self, ident: Ident, details: Details) {
        self.bindings.make_mut().insert(ident, details);
    }

    pub(crate) fn remove_variable(&mut self, ident: &Ident) -> Option<Details> {
        self.bindings.make_mut().remove(ident)
    }

    /// Any state the child scope modified that was part of the parent is copied to the parent scope
    pub(crate) fn apply_child_scope(mut self, child: Self) -> Self {
        // The child never wrote: every binding it could copy back is the one already here.
        if self.bindings.ptr_eq(&child.bindings) {
            return self;
        }

        for (ident, child_details) in child.bindings.into_map() {
            if let Some(self_details) = self.bindings.make_mut().get_mut(&ident) {
                *self_details = child_details;
            }
        }

        self
    }

    /// Merges two local envs together. This is useful in cases such as if statements
    /// where different `LocalEnv`'s can be created, and the result is decided at runtime.
    /// The compile-time type must be the union of the options.
    pub(crate) fn merge(mut self, other: Self) -> Self {
        // Neither side wrote since the fork, so every binding would be merged with itself.
        // `Details::merge` is idempotent (`TypeDef::union` of equal type defs, and equal
        // values are kept), so the whole merge is a no-op. This also avoids the
        // `Arc::make_mut` deep copy of the binding map that the loop would otherwise force.
        if self.bindings.ptr_eq(&other.bindings) {
            return self;
        }

        for (ident, other_details) in other.bindings.into_map() {
            let bindings = self.bindings.make_mut();
            if let Some(self_details) = bindings.get_mut(&ident) {
                *self_details = self_details.clone().merge(other_details);
            } else {
                bindings.insert(ident, other_details);
            }
        }
        self
    }
}

/// A lexical scope within the program.
#[derive(Debug, Clone)]
pub struct ExternalEnv {
    /// The external target of the program.
    target: Details,

    /// The type of metadata
    metadata: Kind,
}

impl Default for ExternalEnv {
    fn default() -> Self {
        Self::new_with_kind(
            Kind::object(Collection::any()),
            Kind::object(Collection::any()),
        )
    }
}

impl ExternalEnv {
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            target: self.target.merge(other.target),
            metadata: self.metadata.union(other.metadata),
        }
    }

    /// Creates a new external environment that starts with an initial given
    /// [`Kind`].
    #[must_use]
    pub fn new_with_kind(target: Kind, metadata: Kind) -> Self {
        Self {
            target: Details {
                type_def: target.into(),
                value: None,
            },
            metadata,
        }
    }

    pub(crate) fn target(&self) -> &Details {
        &self.target
    }

    pub(crate) fn target_mut(&mut self) -> &mut Details {
        &mut self.target
    }

    pub fn target_kind(&self) -> &Kind {
        self.target().type_def.kind()
    }

    pub fn kind(&self, prefix: PathPrefix) -> Kind {
        match prefix {
            PathPrefix::Event => self.target_kind(),
            PathPrefix::Metadata => self.metadata_kind(),
        }
        .clone()
    }

    pub fn metadata_kind(&self) -> &Kind {
        &self.metadata
    }

    pub(crate) fn metadata_kind_mut(&mut self) -> &mut Kind {
        &mut self.metadata
    }

    pub(crate) fn update_target(&mut self, details: Details) {
        self.target = details;
    }

    pub fn update_metadata(&mut self, kind: Kind) {
        self.metadata = kind;
    }
}

/// The state used at runtime to track changes as they happen.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Default)]
pub struct RuntimeState {
    /// The [`Value`] stored in each variable.
    variables: HashMap<Ident, Value>,
}

impl RuntimeState {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    pub fn clear(&mut self) {
        self.variables.clear();
    }

    #[must_use]
    pub fn variable(&self, ident: &Ident) -> Option<&Value> {
        self.variables.get(ident)
    }

    pub fn variable_mut(&mut self, ident: &Ident) -> Option<&mut Value> {
        self.variables.get_mut(ident)
    }

    pub(crate) fn insert_variable(&mut self, ident: Ident, value: Value) {
        self.variables.insert(ident, value);
    }

    pub(crate) fn remove_variable(&mut self, ident: &Ident) {
        self.variables.remove(ident);
    }

    pub(crate) fn swap_variable(&mut self, ident: Ident, value: Value) -> Option<Value> {
        match self.variables.entry(ident) {
            Entry::Occupied(mut v) => Some(std::mem::replace(v.get_mut(), value)),
            Entry::Vacant(v) => {
                v.insert(value);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn details(type_def: TypeDef, value: Option<Value>) -> Details {
        Details { type_def, value }
    }

    /// Two bindings whose kinds are deep enough that a merge is observable.
    fn env(bar: TypeDef) -> LocalEnv {
        let mut env = LocalEnv::default();
        env.insert_variable(
            Ident::new("foo"),
            details(
                TypeDef::object(Collection::from_parts(
                    [("a".into(), Kind::integer())].into(),
                    Kind::bytes(),
                )),
                Some(Value::from(1)),
            ),
        );
        env.insert_variable(Ident::new("bar"), details(bar, None));
        env
    }

    /// `LocalEnv::merge` short-circuits when both sides still share the binding map. The
    /// result has to be what merging a *distinct but equal* env produces.
    #[test]
    fn merge_shared_bindings_matches_distinct_equal_env() {
        let this = env(TypeDef::bytes());

        // Same allocation (the short-circuit fires).
        let shared = this.clone();
        assert!(this.bindings.ptr_eq(&shared.bindings));

        // Equal by value, separate allocation (the short-circuit cannot fire).
        let distinct = env(TypeDef::bytes());
        assert!(!this.bindings.ptr_eq(&distinct.bindings));

        assert_eq!(this.clone().merge(shared), this.clone().merge(distinct));
        assert_eq!(this.clone().merge(env(TypeDef::bytes())), this);
    }

    /// The short-circuit must not fire once either side has written.
    #[test]
    fn merge_diverged_bindings_still_unions() {
        let this = env(TypeDef::bytes());
        let mut other = this.clone();
        other.insert_variable(Ident::new("bar"), details(TypeDef::integer(), None));
        other.insert_variable(Ident::new("baz"), details(TypeDef::null(), None));

        let merged = this.merge(other);

        assert_eq!(
            merged.variable(&Ident::new("bar")).unwrap().type_def,
            TypeDef::bytes().or_integer()
        );
        assert_eq!(
            merged.variable(&Ident::new("baz")).unwrap().type_def,
            TypeDef::null()
        );
    }

    #[test]
    fn apply_child_scope_shared_matches_distinct_equal_env() {
        let this = env(TypeDef::bytes());
        let shared = this.clone();
        let distinct = env(TypeDef::bytes());

        assert_eq!(
            this.clone().apply_child_scope(shared),
            this.clone().apply_child_scope(distinct)
        );
        assert_eq!(this.clone().apply_child_scope(env(TypeDef::bytes())), this);
    }

    /// A child that did write still copies its updates back into the parent.
    #[test]
    fn apply_child_scope_diverged_child_overwrites() {
        let this = env(TypeDef::bytes());
        let mut child = this.clone();
        child.insert_variable(Ident::new("bar"), details(TypeDef::integer(), None));
        child.insert_variable(Ident::new("scoped"), details(TypeDef::null(), None));

        let applied = this.apply_child_scope(child);

        assert_eq!(
            applied.variable(&Ident::new("bar")).unwrap().type_def,
            TypeDef::integer()
        );
        assert!(applied.variable(&Ident::new("scoped")).is_none());
    }

    /// `TypeState::merge` of a state with a shared clone of itself is the identity.
    #[test]
    fn type_state_merge_with_shared_clone_is_identity() {
        let state = TypeState {
            local: env(TypeDef::bytes()),
            external: ExternalEnv::new_with_kind(
                Kind::object(Collection::from_parts(
                    [("a".into(), Kind::integer())].into(),
                    Kind::bytes(),
                )),
                Kind::object(Collection::any()),
            ),
        };

        let merged = state.clone().merge(state.clone());

        assert_eq!(merged.local, state.local);
        assert_eq!(merged.external.target(), state.external.target());
        assert_eq!(
            merged.external.metadata_kind(),
            state.external.metadata_kind()
        );
    }
}
