//! Module containing constraint based algorithms like CTPC and Hiton.

use itertools::Itertools;
use std::collections::BTreeSet;
use std::usize;

use super::hypothesis_test::*;
use crate::parameter_learning::{Cache, ParameterLearning};
use crate::structure_learning::StructureLearningAlgorithm;
use crate::{process, tools};

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
    fn fit_transform<T>(&mut self, net: T, dataset: &tools::Dataset) -> T
    where
        T: process::NetworkProcess,
    {
        //Check the coherence between dataset and network
        if net.get_number_of_nodes() != dataset.get_trajectories()[0].get_events().shape()[1] {
            panic!("Dataset and Network must have the same number of variables.")
        }

        //Make the network mutable.
        let mut net = net;

        net.initialize_adj_matrix();

        for child_node in net.get_node_indices() {
            let mut candidate_parent_set: BTreeSet<usize> = net
                .get_node_indices()
                .into_iter()
                .filter(|x| x != &child_node)
                .collect();
            let mut b = 0;
            while b < candidate_parent_set.len() {
                for parent_node in candidate_parent_set.iter() {
                    for separation_set in candidate_parent_set
                        .iter()
                        .filter(|x| x != &parent_node)
                        .map(|x| *x)
                        .combinations(b)
                    {
                        let separation_set = separation_set.into_iter().collect();
                        if self.Ftest.call(
                            &net,
                            child_node,
                            *parent_node,
                            &separation_set,
                            &mut self.cache,
                        ) && self.Chi2test.call(&net, child_node, *parent_node, &separation_set, &mut self.cache) {
                            candidate_parent_set.remove(&parent_node);
                            break;
                        }
                    }
                }
                b = b + 1;
            }
        }

        net
    }
}
