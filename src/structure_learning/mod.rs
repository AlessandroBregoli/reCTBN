pub mod score_function;
pub mod score_based_algorithm;
use crate::network;
use crate::tools;

pub trait StructureLearningAlgorithm {
    fn call<T, >(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network;
}
