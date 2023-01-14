//! Module containing constraint based algorithms like CTPC and Hiton.

use crate::params::Params;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::ParallelExtend;
use std::collections::{BTreeSet, HashMap};
use std::mem;
use std::usize;

use super::hypothesis_test::*;
use crate::parameter_learning::ParameterLearning;
use crate::process;
use crate::structure_learning::StructureLearningAlgorithm;
use crate::tools::Dataset;

pub struct Cache<'a, P: ParameterLearning> {
    parameter_learning: &'a P,
    cache_persistent_small: HashMap<Option<BTreeSet<usize>>, Params>,
    cache_persistent_big: HashMap<Option<BTreeSet<usize>>, Params>,
    parent_set_size_small: usize,
}

impl<'a, P: ParameterLearning> Cache<'a, P> {
    pub fn new(parameter_learning: &'a P) -> Cache<'a, P> {
        Cache {
            parameter_learning,
            cache_persistent_small: HashMap::new(),
            cache_persistent_big: HashMap::new(),
            parent_set_size_small: 0,
        }
    }
    pub fn fit<T: process::NetworkProcess>(
        &mut self,
        net: &T,
        dataset: &Dataset,
        node: usize,
        parent_set: Option<BTreeSet<usize>>,
    ) -> Params {
        let parent_set_len = parent_set.as_ref().unwrap().len();
        if parent_set_len > self.parent_set_size_small + 1 {
            //self.cache_persistent_small = self.cache_persistent_big;
            mem::swap(
                &mut self.cache_persistent_small,
                &mut self.cache_persistent_big,
            );
            self.cache_persistent_big = HashMap::new();
            self.parent_set_size_small += 1;
        }

        if parent_set_len > self.parent_set_size_small {
            match self.cache_persistent_big.get(&parent_set) {
                // TODO: Better not clone `params`, useless clock cycles, RAM use and I/O
                // not cloning requires a minor and reasoned refactoring across the library
                Some(params) => params.clone(),
                None => {
                    let params =
                        self.parameter_learning
                            .fit(net, dataset, node, parent_set.clone());
                    self.cache_persistent_big.insert(parent_set, params.clone());
                    params
                }
            }
        } else {
            match self.cache_persistent_small.get(&parent_set) {
                // TODO: Better not clone `params`, useless clock cycles, RAM use and I/O
                // not cloning requires a minor and reasoned refactoring across the library
                Some(params) => params.clone(),
                None => {
                    let params =
                        self.parameter_learning
                            .fit(net, dataset, node, parent_set.clone());
                    self.cache_persistent_small
                        .insert(parent_set, params.clone());
                    params
                }
            }
        }
    }
}

pub struct CTPC<P: ParameterLearning> {
    parameter_learning: P,
    Ftest: F,
    Chi2test: ChiSquare,
}

impl<P: ParameterLearning> CTPC<P> {
    pub fn new(parameter_learning: P, Ftest: F, Chi2test: ChiSquare) -> CTPC<P> {
        CTPC {
            parameter_learning,
            Ftest,
            Chi2test,
        }
    }
}

impl<P: ParameterLearning> StructureLearningAlgorithm for CTPC<P> {
    fn fit_transform<T>(&self, net: T, dataset: &Dataset) -> T
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

        let mut learned_parent_sets: Vec<(usize, BTreeSet<usize>)> = vec![];
        learned_parent_sets.par_extend(net.get_node_indices().into_par_iter().map(|child_node| {
            let mut cache = Cache::new(&self.parameter_learning);
            let mut candidate_parent_set: BTreeSet<usize> = net
                .get_node_indices()
                .into_iter()
                .filter(|x| x != &child_node)
                .collect();
            let mut separation_set_size = 0;
            while separation_set_size < candidate_parent_set.len() {
                let mut candidate_parent_set_TMP = candidate_parent_set.clone();
                for parent_node in candidate_parent_set.iter() {
                    for separation_set in candidate_parent_set
                        .iter()
                        .filter(|x| x != &parent_node)
                        .map(|x| *x)
                        .combinations(separation_set_size)
                    {
                        let separation_set = separation_set.into_iter().collect();
                        if self.Ftest.call(
                            &net,
                            child_node,
                            *parent_node,
                            &separation_set,
                            dataset,
                            &mut cache,
                        ) && self.Chi2test.call(
                            &net,
                            child_node,
                            *parent_node,
                            &separation_set,
                            dataset,
                            &mut cache,
                        ) {
                            candidate_parent_set_TMP.remove(parent_node);
                            break;
                        }
                    }
                }
                candidate_parent_set = candidate_parent_set_TMP;
                separation_set_size += 1;
            }
            (child_node, candidate_parent_set)
        }));
        for (child_node, candidate_parent_set) in learned_parent_sets {
            for parent_node in candidate_parent_set.iter() {
                net.add_edge(*parent_node, child_node);
            }
        }
        net
    }
}
