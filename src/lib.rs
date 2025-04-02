//! [`WeakMap`] is a hash map that stores weak references to values.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

pub mod map;
pub use map::WeakMap;

mod traits;
pub use traits::{StrongRef, WeakRef};
