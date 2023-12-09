//! Learn the structure of the network.

pub mod constraint_based_algorithm;
pub mod hypothesis_test;
pub mod score_based_algorithm;
pub mod score_function;
use crate::{process, tools::Dataset};

/// It defines the required methods for a _structure learning algorithm_.
pub trait StructuralLearningAlgorithm {
    /// Learn the structure of a network
    ///
    /// #Arguments
    ///
    /// * `net`: a `NetworkProcess` instance
    /// * `dataset`: instantiation of the `struct tools::Dataset` containing the
    ///              observations used to learn the struct.
    ///
    /// # Return
    ///
    /// * Return a `NetworkProcess` with the learned structure.
    fn fit_transform<T>(&self, net: T, dataset: &Dataset) -> T
    where
        T: process::NetworkProcess;
}
