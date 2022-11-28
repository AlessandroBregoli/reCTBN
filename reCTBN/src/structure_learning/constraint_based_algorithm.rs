//! Module containing constraint based algorithms like CTPC and Hiton.

use super::hypothesis_test::*;
use crate::structure_learning::StructureLearningAlgorithm;
use crate::{process, tools};
use crate::parameter_learning::{Cache, ParameterLearning};

pub struct CTPC<P: ParameterLearning> {
    Ftest: F,
    Chi2test: ChiSquare,
    cache: Cache<P>,

}

impl<P: ParameterLearning> CTPC<P> {
    pub fn new(Ftest: F, Chi2test: ChiSquare, cache: Cache<P>) -> CTPC<P> {
        CTPC {
            Chi2test,
            Ftest,
            cache,
        }
    }
}

impl<P: ParameterLearning> StructureLearningAlgorithm for CTPC<P> {
    fn fit_transform<T>(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: process::NetworkProcess,
    {
        //Check the coherence between dataset and network
        if net.get_number_of_nodes() != dataset.get_trajectories()[0].get_events().shape()[1] {
            panic!("Dataset and Network must have the same number of variables.")
        }

        //Make the network mutable.
        let mut net = net;
        net
    }
}
