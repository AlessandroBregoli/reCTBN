pub mod score_function;
pub mod score_based_algorithm;
use crate::network;
use crate::tools;

pub trait StructureLearningAlgorithm {
    fn fit<T, >(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network;
}
