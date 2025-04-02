//! A hash map that stores weak references to values.

use core::{
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    sync::atomic::{AtomicUsize, Ordering},
};

use hashbrown::{DefaultHashBuilder, Equivalent, TryReserveError, hash_map};

use crate::{StrongRef, WeakRef};

#[derive(Default)]
struct OpsCounter(AtomicUsize);

const OPS_THRESHOLD: usize = 1000;

impl OpsCounter {
    #[inline]
    const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    #[inline]
    fn add(&self, ops: usize) {
        self.0.fetch_add(ops, Ordering::Relaxed);
    }

    #[inline]
    fn bump(&self) {
        self.add(1);
    }

    #[inline]
    fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }

    #[inline]
    fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    #[inline]
    fn reach_threshold(&self) -> bool {
        self.get() >= OPS_THRESHOLD
    }
}

impl Clone for OpsCounter {
    #[inline]
    fn clone(&self) -> Self {
        Self(AtomicUsize::new(self.get()))
    }
}

/// A hash map that stores strong references to values.
pub type StrongMap<K, V, S = DefaultHashBuilder> = hash_map::HashMap<K, V, S>;

/// A hash map that stores weak references to values.
#[derive(Clone)]
pub struct WeakMap<K, V, S = DefaultHashBuilder> {
    inner: hash_map::HashMap<K, V, S>,
    ops: OpsCounter,
}

impl<K, V, S: Default> Default for WeakMap<K, V, S> {
    #[inline]
    fn default() -> Self {
        WeakMap {
            inner: Default::default(),
            ops: Default::default(),
        }
    }
}

impl<K, V> WeakMap<K, V, DefaultHashBuilder> {
    /// Creates an empty `WeakMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not
    /// allocate until it is first inserted into.
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: hash_map::HashMap::new(),
            ops: OpsCounter::new(),
        }
    }

    /// Creates an empty `WeakMap` with the specified capacity.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: hash_map::HashMap::with_capacity(capacity),
            ops: OpsCounter::new(),
        }
    }
}

impl<K, V, S> WeakMap<K, V, S> {
    /// Creates an empty `WeakMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not
    /// allocate until it is first inserted into.
    #[inline]
    pub const fn with_hasher(hasher: S) -> Self {
        Self {
            inner: hash_map::HashMap::with_hasher(hasher),
            ops: OpsCounter::new(),
        }
    }

    /// Creates an empty `WeakMap` with the specified capacity, using
    /// `hash_builder` to hash the keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            inner: hash_map::HashMap::with_capacity_and_hasher(capacity, hash_builder),
            ops: OpsCounter::new(),
        }
    }
}

impl<K, V, S> From<hash_map::HashMap<K, V, S>> for WeakMap<K, V, S> {
    #[inline]
    fn from(inner: hash_map::HashMap<K, V, S>) -> Self {
        Self {
            inner,
            ops: OpsCounter::new(),
        }
    }
}

impl<K, V, S> From<WeakMap<K, V, S>> for hash_map::HashMap<K, V, S> {
    #[inline]
    fn from(map: WeakMap<K, V, S>) -> Self {
        map.inner
    }
}

impl<K, V, S> WeakMap<K, V, S> {
    /// Returns a reference to the map's [`BuildHasher`].
    ///
    /// [`BuildHasher`]: https://doc.rust-lang.org/std/hash/trait.BuildHasher.html
    #[inline]
    pub fn hasher(&self) -> &S {
        self.inner.hasher()
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the `WeakMap` might be able to hold more,
    /// but is guaranteed to be able to hold at least this many.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.ops.reset();
    }

    /// Returns the number of elements in the underlying map.
    #[must_use]
    pub fn raw_len(&self) -> usize {
        self.inner.len()
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    #[inline]
    pub fn iter(&self) -> Iter<K, V> {
        let it = self.inner.iter();
        self.ops.add(it.len());
        Iter(it)
    }

    /// An iterator visiting all keys in arbitrary order.
    #[inline]
    pub fn keys(&self) -> Keys<K, V> {
        Keys(self.iter())
    }

    /// Creates a consuming iterator visiting all the keys in arbitrary order.
    /// The map cannot be used after calling this.
    #[inline]
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys(IntoIter(self.inner.into_iter()))
    }

    /// An iterator visiting all values in arbitrary order.
    #[inline]
    pub fn values(&self) -> Values<K, V> {
        Values(self.iter())
    }

    /// Creates a consuming iterator visiting all the values in arbitrary order.
    /// The map cannot be used after calling this.
    #[inline]
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues(IntoIter(self.inner.into_iter()))
    }
}

impl<K, V, S> WeakMap<K, V, S>
where
    V: WeakRef,
{
    #[inline]
    fn cleanup(&mut self) {
        self.ops.reset();
        self.inner.retain(|_, v| !v.is_expired());
    }

    #[inline]
    fn try_bump(&mut self) {
        self.ops.bump();
        if self.ops.reach_threshold() {
            self.cleanup();
        }
    }

    /// Returns the number of elements in the map, excluding expired values.
    ///
    /// This is a linear operation, as it iterates over all elements in the map.
    ///
    /// The returned value may be less than the result of [`Self::raw_len`].
    #[inline]
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Returns `true` if the map contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// If the returned iterator is dropped before being fully consumed, it
    /// drops the remaining key-value pairs. The returned iterator keeps a
    /// mutable borrow on the vector to optimize its implementation.
    #[inline]
    pub fn drain(&mut self) -> Drain<K, V> {
        self.cleanup();
        Drain(self.inner.drain())
    }

    /// Retains only the elements specified by the predicate. Keeps the
    /// allocated memory for reuse.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, V::Strong) -> bool,
    {
        self.ops.reset();
        self.inner.retain(|k, v| {
            if let Some(v) = v.upgrade() {
                f(k, v)
            } else {
                false
            }
        });
    }
}

impl<K, V, S> WeakMap<K, V, S>
where
    K: Eq + Hash,
    V: WeakRef,
    S: BuildHasher,
{
    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `WeakMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    pub fn reserve(&mut self, additional: usize) {
        self.cleanup();
        self.inner.reserve(additional);
    }

    /// Tries to reserve capacity for at least `additional` more elements to be
    /// inserted in the `WeakMap`. The collection may reserve more space
    /// to avoid frequent reallocations.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.cleanup();
        self.inner.try_reserve(additional)
    }

    /// Shrinks the capacity of the map as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    pub fn shrink_to_fit(&mut self) {
        self.cleanup();
        self.inner.shrink_to_fit();
    }

    /// Shrinks the capacity of the map with a lower limit. It will drop
    /// down no lower than the supplied limit while maintaining the internal
    /// rules and possibly leaving some space in accordance with the resize
    /// policy.
    ///
    /// This function does nothing if the current capacity is smaller than the
    /// supplied minimum capacity.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.cleanup();
        self.inner.shrink_to(min_capacity);
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    pub fn get<Q>(&self, key: &Q) -> Option<V::Strong>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.ops.bump();
        self.inner.get(key).and_then(V::upgrade)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, V::Strong)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.ops.bump();
        self.inner
            .get_key_value(key)
            .and_then(|(k, v)| v.upgrade().map(|v| (k, v)))
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.ops.bump();
        self.inner.get(key).is_some_and(|v| !v.is_expired())
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the
    /// [`std::collections`] [module-level documentation] for more.
    ///
    /// [`None`]: https://doc.rust-lang.org/std/option/enum.Option.html#variant.None
    /// [`std::collections`]: https://doc.rust-lang.org/std/collections/index.html
    /// [module-level documentation]: https://doc.rust-lang.org/std/collections/index.html#insert-and-complex-keys
    pub fn insert(&mut self, key: K, value: &V::Strong) -> Option<V::Strong> {
        self.try_bump();
        self.inner
            .insert(key, V::Strong::downgrade(value))
            .and_then(|v| v.upgrade())
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map. Keeps the allocated memory for reuse.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V::Strong>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.try_bump();
        self.inner.remove(key).and_then(|v| v.upgrade())
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map. Keeps the allocated memory for reuse.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: https://doc.rust-lang.org/std/cmp/trait.Eq.html
    /// [`Hash`]: https://doc.rust-lang.org/std/hash/trait.Hash.html
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V::Strong)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.try_bump();
        self.inner
            .remove_entry(key)
            .and_then(|(k, v)| v.upgrade().map(|v| (k, v)))
    }

    /// Returns the total amount of memory allocated internally by the hash
    /// set, in bytes.
    ///
    /// The returned number is informational only. It is intended to be
    /// primarily used for memory profiling.
    #[inline]
    pub fn allocation_size(&self) -> usize {
        self.inner.allocation_size()
    }

    /// Upgrade this `WeakMap` to a `StrongMap`.
    pub fn upgrade(&self) -> StrongMap<K, V::Strong, S>
    where
        K: Clone,
        S: Clone,
    {
        self.ops.bump();
        let mut map = StrongMap::with_hasher(self.hasher().clone());
        for (key, value) in self.iter() {
            map.insert(key.clone(), value);
        }
        map
    }
}

impl<K, V, S> PartialEq for WeakMap<K, V, S>
where
    K: Eq + Hash,
    V: WeakRef,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter().all(|(key, value)| {
            other
                .get(key)
                .is_some_and(|v| V::Strong::ptr_eq(&value, &v))
        })
    }
}

impl<K, V, S> Eq for WeakMap<K, V, S>
where
    K: Eq + Hash,
    V: WeakRef,
    S: BuildHasher,
{
}

impl<K, V, S> fmt::Debug for WeakMap<K, V, S>
where
    K: fmt::Debug,
    V: WeakRef,
    V::Strong: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<'a, K, V, S> FromIterator<(K, &'a V::Strong)> for WeakMap<K, V, S>
where
    K: Eq + Hash,
    V: WeakRef,
    S: BuildHasher + Default,
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, &'a V::Strong)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut map = WeakMap::with_capacity_and_hasher(iter.size_hint().0, S::default());
        for (key, value) in iter {
            map.insert(key, value);
        }
        map
    }
}

impl<K, V, S, const N: usize> From<[(K, &V::Strong); N]> for WeakMap<K, V, S>
where
    K: Eq + Hash,
    V: WeakRef,
    S: BuildHasher + Default,
{
    #[inline]
    fn from(array: [(K, &V::Strong); N]) -> Self {
        array.into_iter().collect()
    }
}

impl<K, V, S> From<&StrongMap<K, V::Strong, S>> for WeakMap<K, V, S>
where
    K: Eq + Hash + Clone,
    V: WeakRef,
    S: BuildHasher + Clone,
{
    fn from(value: &StrongMap<K, V::Strong, S>) -> Self {
        let mut map = WeakMap::with_capacity_and_hasher(value.len(), value.hasher().clone());
        for (key, value) in value.iter() {
            map.insert(key.clone(), value);
        }
        map
    }
}

/// An iterator over the entries of a `HashMap` in arbitrary order.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Iter<'a, K, V>(hash_map::Iter<'a, K, V>);

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    V: WeakRef,
{
    type Item = (&'a K, V::Strong);

    fn next(&mut self) -> Option<Self::Item> {
        for (key, value) in self.0.by_ref() {
            if let Some(value) = value.upgrade() {
                return Some((key, value));
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.len()))
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> where V: WeakRef {}

impl<K, V> Default for Iter<'_, K, V> {
    fn default() -> Self {
        Iter(hash_map::Iter::default())
    }
}

impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter(self.0.clone())
    }
}

impl<K, V> fmt::Debug for Iter<'_, K, V>
where
    K: fmt::Debug,
    V: WeakRef,
    V::Strong: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, K, V> IntoIterator for &'a WeakMap<K, V>
where
    V: WeakRef,
{
    type IntoIter = Iter<'a, K, V>;
    type Item = (&'a K, V::Strong);

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator over the keys of a `WeakMap`.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Keys<'a, K, V>(Iter<'a, K, V>);

impl<'a, K, V> Iterator for Keys<'a, K, V>
where
    V: WeakRef,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> FusedIterator for Keys<'_, K, V> where V: WeakRef {}

impl<K, V> Default for Keys<'_, K, V> {
    fn default() -> Self {
        Keys(Iter::default())
    }
}

impl<K, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Keys(self.0.clone())
    }
}

impl<K, V> fmt::Debug for Keys<'_, K, V>
where
    K: fmt::Debug,
    V: WeakRef,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An iterator over the values of a `WeakMap`.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Values<'a, K, V>(Iter<'a, K, V>);

impl<K, V> Iterator for Values<'_, K, V>
where
    V: WeakRef,
{
    type Item = V::Strong;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> FusedIterator for Values<'_, K, V> where V: WeakRef {}

impl<K, V> Default for Values<'_, K, V> {
    fn default() -> Self {
        Values(Iter::default())
    }
}

impl<K, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Values(self.0.clone())
    }
}

impl<K, V> fmt::Debug for Values<'_, K, V>
where
    V: WeakRef,
    V::Strong: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

/// An owning iterator over the entries of a `HashMap` in arbitrary order.
pub struct IntoIter<K, V>(hash_map::IntoIter<K, V>);

impl<K, V> Iterator for IntoIter<K, V>
where
    V: WeakRef,
{
    type Item = (K, V::Strong);

    fn next(&mut self) -> Option<Self::Item> {
        for (key, value) in self.0.by_ref() {
            if let Some(value) = value.upgrade() {
                return Some((key, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.len()))
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> where V: WeakRef {}

impl<K, V> Default for IntoIter<K, V> {
    fn default() -> Self {
        IntoIter(hash_map::IntoIter::default())
    }
}

impl<K, V> IntoIterator for WeakMap<K, V>
where
    V: WeakRef,
{
    type IntoIter = IntoIter<K, V>;
    type Item = (K, V::Strong);

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.inner.into_iter())
    }
}

/// An owning iterator over the keys of a `HashMap` in arbitrary order.
pub struct IntoKeys<K, V>(IntoIter<K, V>);

impl<K, V> Iterator for IntoKeys<K, V>
where
    V: WeakRef,
{
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(key, _)| key)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> FusedIterator for IntoKeys<K, V> where V: WeakRef {}

impl<K, V> Default for IntoKeys<K, V> {
    fn default() -> Self {
        IntoKeys(IntoIter::default())
    }
}

/// An owning iterator over the values of a `HashMap` in arbitrary order.
pub struct IntoValues<K, V>(IntoIter<K, V>);

impl<K, V> Iterator for IntoValues<K, V>
where
    V: WeakRef,
{
    type Item = V::Strong;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, value)| value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> FusedIterator for IntoValues<K, V> where V: WeakRef {}

impl<K, V> Default for IntoValues<K, V> {
    fn default() -> Self {
        IntoValues(IntoIter::default())
    }
}
/// A draining iterator over the entries of a `HashMap` in arbitrary
/// order.
pub struct Drain<'a, K, V>(hash_map::Drain<'a, K, V>);

impl<K, V> Iterator for Drain<'_, K, V>
where
    V: WeakRef,
{
    type Item = (K, V::Strong);

    fn next(&mut self) -> Option<Self::Item> {
        for (key, value) in self.0.by_ref() {
            if let Some(value) = value.upgrade() {
                return Some((key, value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.len()))
    }
}

impl<K, V> FusedIterator for Drain<'_, K, V> where V: WeakRef {}

#[cfg(test)]
mod tests {
    use alloc::sync::{Arc, Weak};

    use super::*;

    #[test]
    fn test_basic() {
        let mut map = WeakMap::<u32, Weak<&str>>::new();

        let elem1 = Arc::new("1");
        map.insert(1, &elem1);

        {
            let elem2 = Arc::new("2");
            map.insert(2, &elem2);
        }

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(elem1));
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn test_cleanup() {
        let mut map = WeakMap::<usize, Weak<usize>>::new();

        for i in 0..OPS_THRESHOLD * 10 {
            let elem = Arc::new(i);
            map.insert(i, &elem);
        }

        assert_eq!(map.len(), 0);
        assert_eq!(map.raw_len(), 1);
    }
}
