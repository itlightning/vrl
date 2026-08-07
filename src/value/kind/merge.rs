//! All types related to merging one [`Kind`] into another.

use super::Field;
use std::ops::BitOr;

use super::{Collection, Kind};

impl Kind {
    /// Merge `other` into `self`, using the provided `Strategy`.
    pub fn merge(&mut self, other: Self, strategy: Strategy) {
        self.merge_keep(other, strategy.collisions.is_shallow());
    }

    fn merge_primitives(&mut self, other: &Self) {
        self.bytes = self.bytes.or(other.bytes);
        self.integer = self.integer.or(other.integer);
        self.float = self.float.or(other.float);
        self.boolean = self.boolean.or(other.boolean);
        self.timestamp = self.timestamp.or(other.timestamp);
        self.regex = self.regex.or(other.regex);
        self.null = self.null.or(other.null);
        self.undefined = self.undefined.or(other.undefined);
    }

    fn merge_objects(&mut self, other: Option<Box<Collection<Field>>>, overwrite: bool) {
        match (self.object.as_mut(), other) {
            (None, rhs @ Some(_)) => self.object = rhs,
            (Some(lhs), Some(rhs)) => lhs.merge(*rhs, overwrite),
            _ => {}
        }
    }

    /// Returns the union of self and other.
    #[must_use]
    pub fn union(&self, other: Self) -> Self {
        let mut kind = self.clone();
        kind.merge_keep(other, false);
        kind
    }

    /// Merge `other` into `self`, optionally overwriting on conflicts.
    // deprecated
    pub fn merge_keep(&mut self, other: Self, overwrite: bool) {
        self.merge_primitives(&other);
        self.merge_objects(other.object, overwrite);

        match (self.array.as_mut(), other.array) {
            (None, Some(rhs)) => self.array = Some(rhs),
            (Some(lhs), Some(rhs)) => lhs.merge(*rhs, overwrite),
            _ => {}
        }
    }
}

/// The strategy to apply to the merge between two `Kind`s.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Strategy {
    /// How to deal with types in a collection if both specify a type.
    /// This only applies to types in a collection (not the root type)
    pub collisions: CollisionStrategy,
}

/// The choice of "depth" to apply when merging two [`Kind`]s.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CollisionStrategy {
    /// Use the 2nd value. This should no longer be used, and will be removed in the future.
    /// Try using `Kind::insert` or a custom function instead.
    Overwrite,

    /// Merge both together
    Union,
}

impl CollisionStrategy {
    /// Check if `shallow` strategy is enabled.
    #[must_use]
    pub const fn is_shallow(self) -> bool {
        matches!(self, Self::Overwrite)
    }

    /// Check if `deep` strategy is enabled.
    #[must_use]
    pub const fn is_deep(self) -> bool {
        matches!(self, Self::Union)
    }
}

impl BitOr for Kind {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_merge() {
        struct TestCase {
            this: Kind,
            other: Kind,
            strategy: Strategy,
            merged: Kind,
        }

        for (
            title,
            TestCase {
                mut this,
                other,
                strategy,
                merged,
            },
        ) in HashMap::from([
            (
                "object field with unknown",
                TestCase {
                    this: Kind::object(Collection::any()),
                    other: Kind::object(BTreeMap::from([("x".into(), Kind::integer())])),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Union,
                    },
                    merged: {
                        let mut collection =
                            Collection::from(BTreeMap::from([("x".into(), Kind::any())]));
                        collection.set_unknown(Kind::any());
                        Kind::object(collection)
                    },
                },
            ),
            (
                "primitives shallow",
                TestCase {
                    this: Kind::bytes(),
                    other: Kind::integer(),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Overwrite,
                    },
                    merged: Kind::bytes().or_integer(),
                },
            ),
            (
                "primitives deep",
                TestCase {
                    this: Kind::bytes(),
                    other: Kind::integer(),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Union,
                    },
                    merged: Kind::bytes().or_integer(),
                },
            ),
            (
                "mixed unknown shallow",
                TestCase {
                    this: Kind::bytes().or_object(Collection::from_unknown(Kind::integer())),
                    other: Kind::bytes().or_object(Collection::from_unknown(Kind::bytes())),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Overwrite,
                    },
                    merged: Kind::bytes()
                        .or_object(Collection::from_unknown(Kind::integer().or_bytes())),
                },
            ),
            (
                "mixed unknown deep",
                TestCase {
                    this: Kind::bytes().or_object(Collection::from_unknown(Kind::integer())),
                    other: Kind::bytes().or_object(Collection::from_unknown(Kind::bytes())),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Union,
                    },
                    merged: Kind::bytes()
                        .or_object(Collection::from_unknown(Kind::integer().or_bytes())),
                },
            ),
            (
                "mixed known shallow",
                TestCase {
                    this: Kind::bytes().or_object(BTreeMap::from([
                        (
                            "foo".into(),
                            Kind::object(BTreeMap::from([
                                ("qux".into(), Kind::bytes()),
                                ("quux".into(), Kind::boolean()),
                                ("this".into(), Kind::timestamp()),
                            ])),
                        ),
                        ("bar".into(), Kind::integer()),
                    ])),
                    other: Kind::integer().or_object(BTreeMap::from([
                        (
                            "foo".into(),
                            Kind::object(BTreeMap::from([
                                ("qux".into(), Kind::integer()),
                                ("quux".into(), Kind::regex()),
                                ("that".into(), Kind::null()),
                            ])),
                        ),
                        ("baz".into(), Kind::boolean()),
                    ])),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Overwrite,
                    },
                    merged: Kind::bytes().or_integer().or_object(BTreeMap::from([
                        (
                            "foo".into(),
                            Kind::object(BTreeMap::from([
                                ("qux".into(), Kind::integer()),
                                ("quux".into(), Kind::regex()),
                                ("that".into(), Kind::null()),
                            ])),
                        ),
                        ("bar".into(), Kind::integer()),
                        ("baz".into(), Kind::boolean()),
                    ])),
                },
            ),
            (
                "mixed known deep",
                TestCase {
                    this: Kind::bytes().or_object(BTreeMap::from([
                        (
                            "foo".into(),
                            Kind::object(BTreeMap::from([
                                ("qux".into(), Kind::bytes()),
                                ("quux".into(), Kind::boolean()),
                                ("this".into(), Kind::timestamp()),
                            ])),
                        ),
                        ("bar".into(), Kind::integer()),
                    ])),
                    other: Kind::integer().or_object(BTreeMap::from([
                        (
                            "foo".into(),
                            Kind::object(BTreeMap::from([
                                ("qux".into(), Kind::integer()),
                                ("quux".into(), Kind::regex()),
                                ("that".into(), Kind::null()),
                            ])),
                        ),
                        ("baz".into(), Kind::boolean()),
                    ])),
                    strategy: Strategy {
                        collisions: CollisionStrategy::Union,
                    },
                    merged: Kind::bytes().or_integer().or_object(BTreeMap::from([
                        (
                            "foo".into(),
                            Kind::object(BTreeMap::from([
                                ("qux".into(), Kind::bytes().or_integer()),
                                ("quux".into(), Kind::boolean().or_regex()),
                                ("this".into(), Kind::timestamp().or_undefined()),
                                ("that".into(), Kind::null().or_undefined()),
                            ])),
                        ),
                        ("bar".into(), Kind::integer().or_undefined()),
                        ("baz".into(), Kind::boolean().or_undefined()),
                    ])),
                },
            ),
        ]) {
            this.merge(other, strategy);
            assert_eq!(this, merged, "{title}");
        }
    }

    /// A `Kind` with nested objects and arrays, rebuilt from scratch on every call so two
    /// calls are value-equal but share no `Arc`.
    fn nested_kind() -> Kind {
        Kind::bytes()
            .or_object(BTreeMap::from([
                ("a".into(), Kind::integer()),
                (
                    "b".into(),
                    Kind::object(BTreeMap::from([("inner".into(), Kind::bytes())])),
                ),
            ]))
            .or_array(BTreeMap::from([(0.into(), Kind::timestamp())]))
    }

    /// Merging a `Kind` with an `Arc`-shared clone of itself has to give the same answer as
    /// merging it with a value-equal `Kind` that shares nothing, under every strategy.
    #[test]
    fn merge_shared_clone_matches_distinct_equal_kind() {
        for collisions in [CollisionStrategy::Union, CollisionStrategy::Overwrite] {
            let strategy = Strategy { collisions };
            let this = nested_kind();

            let mut from_shared = this.clone();
            from_shared.merge(this.clone(), strategy);

            let mut from_distinct = this.clone();
            from_distinct.merge(nested_kind(), strategy);

            assert_eq!(from_shared, from_distinct, "{collisions:?}");
            assert_eq!(from_shared, this, "{collisions:?}");
        }
    }

    #[test]
    fn union_shared_clone_matches_distinct_equal_kind() {
        let this = nested_kind();

        assert_eq!(this.union(this.clone()), this.union(nested_kind()));
        assert_eq!(this.union(this.clone()), this);
    }

    /// The guards must not swallow a merge that does change something: only one branch of
    /// the tree is shared here, the rest genuinely differs.
    #[test]
    fn merge_partially_shared_kind_still_merges() {
        let shared_branch = Kind::object(BTreeMap::from([("inner".into(), Kind::bytes())]));

        let mut this = Kind::object(BTreeMap::from([
            ("shared".into(), shared_branch.clone()),
            ("own".into(), Kind::integer()),
        ]));
        let other = Kind::object(BTreeMap::from([
            ("shared".into(), shared_branch),
            ("own".into(), Kind::null()),
        ]));

        this.merge_keep(other, false);

        assert_eq!(
            this,
            Kind::object(BTreeMap::from([
                (
                    "shared".into(),
                    Kind::object(BTreeMap::from([("inner".into(), Kind::bytes())]))
                ),
                ("own".into(), Kind::integer().or_null()),
            ]))
        );
    }
}
