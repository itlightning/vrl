use crate::path::OwnedTargetPath;
use crate::value::ValueRegex;
use std::{
    any::{Any, TypeId},
    collections::{BTreeSet, HashMap},
    sync::{Arc, Mutex, PoisonError},
};

type AnyMap = HashMap<TypeId, Box<dyn Any>>;

/// Compiled regex literals, keyed by pattern, shared by every compilation that is
/// handed the same handle.
///
/// A compiled `Regex` is roughly 56 KiB of automata and [`ValueRegex`] is already an
/// `Arc<Regex>`, so identical literals can share one object instead of one each.
///
/// **Scope this to a compile session and no longer.** The cache pins every pattern it
/// has seen, so a process-lifetime handle is a retention bug: patterns from programs
/// that have since been discarded would never be freed. Drop the handle when the
/// session ends, and only the regexes a live `Program` still references survive.
pub type RegexCache = Arc<Mutex<HashMap<String, ValueRegex>>>;

pub struct CompileConfig {
    /// Custom context injected by the external environment
    custom: AnyMap,
    read_only_paths: BTreeSet<ReadOnlyPath>,
    check_unused_expressions: bool,
    /// Shared regex-literal cache, when the caller opted in. `None` compiles every
    /// literal on its own, which is the historical behavior.
    regex_cache: Option<RegexCache>,
}

impl Default for CompileConfig {
    fn default() -> Self {
        CompileConfig {
            custom: AnyMap::default(),
            read_only_paths: BTreeSet::default(),
            check_unused_expressions: true,
            regex_cache: None,
        }
    }
}

impl CompileConfig {
    /// Get external context data from the external environment.
    #[must_use]
    pub fn get_custom<T: 'static>(&self) -> Option<&T> {
        self.custom
            .get(&TypeId::of::<T>())
            .and_then(|t| t.downcast_ref())
    }

    /// Get external context data from the external environment.
    pub fn get_custom_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.custom
            .get_mut(&TypeId::of::<T>())
            .and_then(|t| t.downcast_mut())
    }

    /// Sets the external context data for VRL functions to use.
    pub fn set_custom<T: 'static>(&mut self, data: T) {
        self.custom.insert(TypeId::of::<T>(), Box::new(data) as _);
    }

    pub fn custom_mut(&mut self) -> &mut AnyMap {
        &mut self.custom
    }

    /// Marks everything as read only. Any mutations on read-only values will result in a
    /// compile time error.
    pub fn set_read_only(&mut self) {
        self.set_read_only_path(OwnedTargetPath::event_root(), true);
        self.set_read_only_path(OwnedTargetPath::metadata_root(), true);
    }

    #[must_use]
    pub fn is_read_only_path(&self, path: &OwnedTargetPath) -> bool {
        for read_only_path in &self.read_only_paths {
            // any paths that are a parent of read-only paths also can't be modified
            if read_only_path.path.can_start_with(path) {
                return true;
            }

            if read_only_path.recursive {
                if path.can_start_with(&read_only_path.path) {
                    return true;
                }
            } else if path == &read_only_path.path {
                return true;
            }
        }
        false
    }

    /// Adds a path that is considered read only. Assignments to any paths that match
    /// will fail at compile time.
    pub fn set_read_only_path(&mut self, path: OwnedTargetPath, recursive: bool) {
        self.read_only_paths
            .insert(ReadOnlyPath { path, recursive });
    }

    #[must_use]
    pub fn unused_expression_check_enabled(&self) -> bool {
        self.check_unused_expressions
    }

    pub fn disable_unused_expression_check(&mut self) {
        self.check_unused_expressions = false;
    }

    /// Share one [`RegexCache`] with every other compilation handed the same handle,
    /// so that a pattern appearing in several programs is compiled once.
    ///
    /// The handle must not outlive the compile session. See [`RegexCache`].
    pub fn set_regex_cache(&mut self, cache: RegexCache) {
        self.regex_cache = Some(cache);
    }

    /// The shared regex-literal cache, if one was set.
    #[must_use]
    pub fn regex_cache(&self) -> Option<&RegexCache> {
        self.regex_cache.as_ref()
    }

    /// Compiles a regex literal, reusing the cached automata when this config shares a
    /// [`RegexCache`] with a compilation that already saw the same pattern.
    pub(crate) fn compile_regex_literal(&self, pattern: &str) -> Result<ValueRegex, regex::Error> {
        let Some(cache) = &self.regex_cache else {
            return regex::Regex::new(pattern).map(|regex| ValueRegex::new(Arc::new(regex)));
        };

        // A poisoned cache is still a valid cache: the map is only ever inserted into.
        let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);

        if let Some(regex) = cache.get(pattern) {
            return Ok(regex.clone());
        }

        let regex = ValueRegex::new(Arc::new(regex::Regex::new(pattern)?));
        cache.insert(pattern.to_owned(), regex.clone());

        Ok(regex)
    }
}

#[derive(Debug, Clone, Ord, Eq, PartialEq, PartialOrd)]
struct ReadOnlyPath {
    path: OwnedTargetPath,
    recursive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Eq, Debug)]
    struct Potato(usize);

    #[test]
    fn can_get_custom() {
        let mut config = CompileConfig::default();
        config.set_custom(Potato(42));

        assert_eq!(&Potato(42), config.get_custom::<Potato>().unwrap());
    }

    #[test]
    fn can_get_custom_mut() {
        let mut config = CompileConfig::default();
        config.set_custom(Potato(42));

        let potato = config.get_custom_mut::<Potato>().unwrap();
        potato.0 = 43;

        assert_eq!(&Potato(43), config.get_custom::<Potato>().unwrap());
    }

    #[test]
    fn regex_literals_are_compiled_per_config_by_default() {
        let config = CompileConfig::default();

        let first = config.compile_regex_literal("a+b").unwrap();
        let second = config.compile_regex_literal("a+b").unwrap();

        assert!(!Arc::ptr_eq(&first.into_inner(), &second.into_inner()));
    }

    #[test]
    fn a_shared_cache_compiles_each_pattern_once() {
        let cache = RegexCache::default();
        let mut one = CompileConfig::default();
        let mut two = CompileConfig::default();
        one.set_regex_cache(Arc::clone(&cache));
        two.set_regex_cache(Arc::clone(&cache));

        let from_one = one.compile_regex_literal("a+b").unwrap().into_inner();
        let from_two = two.compile_regex_literal("a+b").unwrap().into_inner();
        let other = two.compile_regex_literal("c+d").unwrap().into_inner();

        assert!(Arc::ptr_eq(&from_one, &from_two));
        assert!(!Arc::ptr_eq(&from_two, &other));
        assert_eq!(2, cache.lock().unwrap().len());
    }

    #[test]
    fn an_invalid_pattern_is_not_cached() {
        let cache = RegexCache::default();
        let mut config = CompileConfig::default();
        config.set_regex_cache(Arc::clone(&cache));

        assert!(config.compile_regex_literal("a(").is_err());
        assert!(cache.lock().unwrap().is_empty());
    }
}
