//! Learn the structure of the network.

pub mod constraint_based_algorithm;
pub mod hypothesis_test;
pub mod score_based_algorithm;
pub mod score_function;
use crate::{network, tools};

pub trait StructureLearningAlgorithm {
    fn fit_transform<T>(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network;
}
