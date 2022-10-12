//! # reCTBN
//!
//! > **Note:** At the moment it's in pre-alpha state. 🧪⚗️💥
//!
//! `reCTBN` is a Continuous Time Bayesian Networks Library written in Rust. 🦀

#![allow(non_snake_case)]
#[cfg(test)]
extern crate approx;

pub mod ctbn;
pub mod network;
pub mod parameter_learning;
pub mod params;
pub mod sampling;
pub mod structure_learning;
pub mod tools;
