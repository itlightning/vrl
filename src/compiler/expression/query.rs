use crate::compiler::{
    Context, Expression,
    expression::{Container, ExpressionError, Resolved, Variable},
    parser::ast::Ident,
    state::ExternalEnv,
    state::{TypeInfo, TypeState},
    type_def::Details,
};
use crate::path::{OwnedTargetPath, OwnedValuePath, PathPrefix};
use crate::value::Value;
use std::borrow::Cow;
use std::fmt;

#[derive(Clone, PartialEq)]
pub struct Query {
    target: Target,
    path: OwnedValuePath,
}

impl Query {
    // TODO:
    // - error when trying to index into object
    // - error when trying to path into array
    #[must_use]
    pub fn new(target: Target, path: OwnedValuePath) -> Self {
        Query { target, path }
    }

    #[must_use]
    pub fn path(&self) -> &OwnedValuePath {
        &self.path
    }

    #[must_use]
    pub fn target(&self) -> &Target {
        &self.target
    }

    #[must_use]
    pub fn is_external(&self) -> bool {
        matches!(self.target, Target::External(_))
    }

    #[must_use]
    pub fn external_path(&self) -> Option<OwnedTargetPath> {
        match self.target {
            Target::External(prefix) => Some(OwnedTargetPath {
                prefix,
                path: self.path.clone(),
            }),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_variable(&self) -> Option<&Variable> {
        match &self.target {
            Target::Internal(variable) => Some(variable),
            _ => None,
        }
    }

    #[must_use]
    pub fn variable_ident(&self) -> Option<&Ident> {
        match &self.target {
            Target::Internal(v) => Some(v.ident()),
            _ => None,
        }
    }

    #[must_use]
    pub fn expression_target(&self) -> Option<&dyn Expression> {
        match &self.target {
            Target::FunctionCall(expr) => Some(expr),
            Target::Container(expr) => Some(expr),
            _ => None,
        }
    }

    // Only "external" paths are supported. Non external paths are ignored
    // see: https://github.com/vectordotdev/vector/issues/11246
    pub fn delete_type_def(&self, external: &mut ExternalEnv, compact: bool) {
        if let Some(target_path) = self.external_path() {
            match target_path.prefix {
                PathPrefix::Event => {
                    let mut type_def = external.target().type_def.clone();
                    type_def.remove(&target_path.path, compact);
                    external.update_target(Details {
                        type_def,
                        value: None,
                    });
                }
                PathPrefix::Metadata => {
                    let mut kind = external.metadata_kind().clone();
                    kind.remove(&target_path.path, compact);
                    external.update_metadata(kind);
                }
            }
        }
    }
}

impl Expression for Query {
    fn resolve(&self, ctx: &mut Context) -> Resolved {
        use Target::{Container, External, FunctionCall, Internal};

        let value = match &self.target {
            External(prefix) => {
                let path = OwnedTargetPath {
                    prefix: *prefix,
                    path: self.path.clone(),
                };
                return Ok(ctx
                    .target()
                    .target_get(&path)
                    .ok()
                    .flatten()
                    .cloned()
                    .unwrap_or(Value::Null));
            }
            // Index the variable in place and clone only the leaf. Resolving
            // the variable first would deep-clone the whole value just to read
            // one field off the copy and drop the rest.
            Internal(variable) => {
                return Ok(ctx
                    .state()
                    .variable(variable.ident())
                    .and_then(|value| value.get(&self.path))
                    .cloned()
                    .unwrap_or(Value::Null));
            }
            FunctionCall(call) => call.resolve(ctx)?,
            Container(container) => container.resolve(ctx)?,
        };

        Ok(value.get(&self.path).cloned().unwrap_or(Value::Null))
    }

    fn resolve_ref<'a>(&self, ctx: &'a mut Context<'_>) -> Result<Cow<'a, Value>, ExpressionError> {
        use Target::{Container, External, FunctionCall, Internal};

        match &self.target {
            External(prefix) => {
                let path = OwnedTargetPath {
                    prefix: *prefix,
                    path: self.path.clone(),
                };
                Ok(ctx
                    .target()
                    .target_get(&path)
                    .ok()
                    .flatten()
                    .map_or(Cow::Owned(Value::Null), Cow::Borrowed))
            }
            Internal(variable) => Ok(ctx
                .state()
                .variable(variable.ident())
                .and_then(|value| value.get(&self.path))
                .map_or(Cow::Owned(Value::Null), Cow::Borrowed)),
            // These produce temporaries: there is nothing outliving the call to
            // borrow from, so they have to materialize.
            FunctionCall(_) | Container(_) => self.resolve(ctx).map(Cow::Owned),
        }
    }

    fn resolve_constant(&self, state: &TypeState) -> Option<Value> {
        match self.target {
            Target::Internal(ref variable) => variable
                .resolve_constant(state)
                .and_then(|v| v.get(self.path()).cloned()),
            _ => None,
        }
    }

    fn type_info(&self, state: &TypeState) -> TypeInfo {
        use Target::{Container, External, FunctionCall, Internal};

        match &self.target {
            External(prefix) => {
                let result = state.external.kind(*prefix).at_path(&self.path).into();
                TypeInfo::new(state, result)
            }
            Internal(variable) => {
                let result = variable.type_def(state).at_path(&self.path);
                TypeInfo::new(state, result)
            }
            FunctionCall(call) => call
                .type_info(state)
                .map_result(|result| result.at_path(&self.path)),
            Container(container) => container
                .type_info(state)
                .map_result(|result| result.at_path(&self.path)),
        }
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.target {
            Target::Internal(_)
                if !self.path.is_root() && !self.path.segments.first().unwrap().is_index() =>
            {
                write!(f, "{}.{}", self.target, self.path)
            }
            _ => write!(f, "{}{}", self.target, self.path),
        }
    }
}

impl fmt::Debug for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Query({:?}, {:?})", self.target, self.path)
    }
}

#[derive(Clone, PartialEq)]
pub enum Target {
    Internal(Variable),
    External(PathPrefix),
    FunctionCall(crate::compiler::expression::FunctionCall),
    Container(Container),
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Target::{Container, External, FunctionCall, Internal};

        match self {
            Internal(v) => v.fmt(f),
            External(prefix) => match prefix {
                PathPrefix::Event => write!(f, "."),
                PathPrefix::Metadata => write!(f, "%"),
            },
            FunctionCall(v) => v.fmt(f),
            Container(v) => v.fmt(f),
        }
    }
}

impl fmt::Debug for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Target::{Container, External, FunctionCall, Internal};

        match self {
            Internal(v) => write!(f, "Internal({v:?})"),
            External(prefix) => match prefix {
                PathPrefix::Event => f.write_str("External(Event)"),
                PathPrefix::Metadata => f.write_str("External(Metadata)"),
            },
            FunctionCall(v) => v.fmt(f),
            Container(v) => v.fmt(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::TargetValueRef;
    use crate::compiler::TimeZone;
    use crate::compiler::state::{LocalEnv, RuntimeState};
    use crate::value::Secrets;
    use std::collections::BTreeMap;

    fn path(path: &str) -> OwnedValuePath {
        path.parse().expect("valid path")
    }

    /// A `Query` over a variable named `var`, bound in the local environment so
    /// that `Variable::new` accepts it.
    fn variable_query(query_path: &str) -> Query {
        let mut local = LocalEnv::default();
        local.insert_variable(
            Ident::new("var"),
            Details {
                type_def: crate::compiler::TypeDef::any(),
                value: None,
            },
        );

        let variable =
            Variable::new((0, 0).into(), Ident::new("var"), &local).expect("variable is bound");

        Query::new(Target::Internal(variable), path(query_path))
    }

    fn container_query(value: Value, query_path: &str) -> Query {
        let container = match crate::compiler::expression::Expr::from(value) {
            crate::compiler::expression::Expr::Container(container) => container,
            other => panic!("expected a container expression, got {other:?}"),
        };

        Query::new(Target::Container(container), path(query_path))
    }

    /// Runs `f` with a context whose event is `event` and whose runtime state
    /// binds `var` to `variable`, when one is given.
    fn with_context<T>(
        event: Value,
        variable: Option<Value>,
        f: impl FnOnce(&mut Context) -> T,
    ) -> T {
        let mut event = event;
        let mut metadata = Value::Object(BTreeMap::new());
        let mut secrets = Secrets::new();
        let mut target = TargetValueRef {
            value: &mut event,
            metadata: &mut metadata,
            secrets: &mut secrets,
        };

        let mut state = RuntimeState::default();
        if let Some(variable) = variable {
            state.insert_variable(Ident::new("var"), variable);
        }

        let timezone = TimeZone::default();
        let mut ctx = Context::new(&mut target, &mut state, &timezone);

        f(&mut ctx)
    }

    fn nested() -> Value {
        Value::from(BTreeMap::from([(
            crate::value::KeyString::from("a"),
            Value::from(BTreeMap::from([(
                crate::value::KeyString::from("b"),
                Value::from(vec![Value::from(1), Value::from(2)]),
            )])),
        )]))
    }

    #[test]
    fn variable_nested_path() {
        let query = variable_query("a.b[1]");
        let got = with_context(Value::Null, Some(nested()), |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(Value::from(2)));
    }

    #[test]
    fn variable_root_path_yields_whole_value() {
        let query = variable_query(".");
        let got = with_context(Value::Null, Some(nested()), |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(nested()));
    }

    #[test]
    fn variable_missing_path_is_null() {
        let query = variable_query("a.nope");
        let got = with_context(Value::Null, Some(nested()), |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(Value::Null));
    }

    #[test]
    fn variable_missing_index_is_null() {
        let query = variable_query("a.b[9]");
        let got = with_context(Value::Null, Some(nested()), |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(Value::Null));
    }

    #[test]
    fn variable_negative_index() {
        let query = variable_query("a.b[-1]");
        let got = with_context(Value::Null, Some(nested()), |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(Value::from(2)));
    }

    #[test]
    fn path_into_scalar_variable_is_null() {
        let query = variable_query("a");
        let got = with_context(Value::Null, Some(Value::from(42)), |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(Value::Null));
    }

    /// The compiler rejects reads of unbound variables, so this is a defensive
    /// path. It must still yield null rather than panicking, for both a root
    /// and a nested query path.
    #[test]
    fn missing_variable_is_null() {
        for query_path in [".", "a.b"] {
            let query = variable_query(query_path);
            let got = with_context(Value::Null, None, |ctx| query.resolve(ctx));

            assert_eq!(got, Ok(Value::Null), "for path {query_path}");
        }
    }

    #[test]
    fn resolve_ref_borrows_from_the_variable() {
        let query = variable_query("a.b[1]");

        with_context(Value::Null, Some(nested()), |ctx| {
            let got = query.resolve_ref(ctx).expect("resolves");
            assert!(
                matches!(got, Cow::Borrowed(_)),
                "expected a borrow of the variable, got {got:?}"
            );
            assert_eq!(got.into_owned(), Value::from(2));
        });
    }

    #[test]
    fn resolve_ref_owns_when_the_path_misses() {
        let query = variable_query("a.nope");

        with_context(Value::Null, Some(nested()), |ctx| {
            let got = query.resolve_ref(ctx).expect("resolves");
            assert!(matches!(got, Cow::Owned(Value::Null)), "got {got:?}");
        });
    }

    #[test]
    fn resolve_ref_borrows_from_the_event() {
        let query = Query::new(Target::External(PathPrefix::Event), path("a.b[1]"));

        with_context(nested(), None, |ctx| {
            let got = query.resolve_ref(ctx).expect("resolves");
            assert!(
                matches!(got, Cow::Borrowed(_)),
                "expected a borrow of the event, got {got:?}"
            );
            assert_eq!(got.into_owned(), Value::from(2));
        });
    }

    /// Containers are temporaries, so they must materialize. Both entry points
    /// have to agree on the value.
    #[test]
    fn container_target_materializes() {
        let query = container_query(nested(), "a.b[1]");

        with_context(Value::Null, None, |ctx| {
            assert_eq!(query.resolve(ctx), Ok(Value::from(2)));

            let got = query.resolve_ref(ctx).expect("resolves");
            assert!(matches!(got, Cow::Owned(_)), "got {got:?}");
            assert_eq!(got.into_owned(), Value::from(2));
        });
    }

    #[test]
    fn container_target_missing_path_is_null() {
        let query = container_query(nested(), "a.nope");
        let got = with_context(Value::Null, None, |ctx| query.resolve(ctx));

        assert_eq!(got, Ok(Value::Null));
    }

    #[test]
    fn test_type_def() {
        let query = Query {
            target: Target::External(PathPrefix::Event),
            path: OwnedValuePath::root(),
        };

        let state = TypeState::default();
        let type_def = query.type_info(&state).result;

        assert!(type_def.is_infallible());
        assert!(type_def.is_object());

        let object = type_def.as_object().unwrap();

        assert!(object.is_any());
    }
}
