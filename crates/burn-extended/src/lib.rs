#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// Alias burn-core as `burn` so derive macros and internal paths match expectations.
use burn_core as burn;

pub mod attention;
pub mod rope;
pub mod sampling;
pub mod generate;
pub mod cache;
pub mod bias;
