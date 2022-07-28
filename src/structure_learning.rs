pub mod score_function;
pub mod score_based_algorithm;
pub mod constraint_based_algorithm;
pub mod hypothesis_test;
use crate::network;
use crate::tools;

pub trait StructureLearningAlgorithm {
    fn fit_transform<T, >(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network;
}
