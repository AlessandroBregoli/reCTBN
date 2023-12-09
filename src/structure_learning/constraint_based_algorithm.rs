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
use crate::structure_learning::StructuralLearningAlgorithm;
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

/// Continuous-Time Peter Clark algorithm.
///
/// A method to learn the structure of the network.
///
/// # Arguments
///
/// * [`parameter_learning`](crate::parameter_learning) - is the method used to learn the parameters.
/// * [`Ftest`](crate::structure_learning::hypothesis_test::F) - is the F-test hyppothesis test.
/// * [`Chi2test`](crate::structure_learning::hypothesis_test::ChiSquare) - is the chi-squared test (χ2 test) hypothesis test.
/// # Example
///
/// ```rust
/// # use std::collections::BTreeSet;
/// # use ndarray::{arr1, arr2, arr3};
/// # use reCTBN::params;
/// # use reCTBN::tools::trajectory_generator;
/// # use reCTBN::process::NetworkProcess;
/// # use reCTBN::process::ctbn::CtbnNetwork;
/// use reCTBN::parameter_learning::BayesianApproach;
/// use reCTBN::structure_learning::StructuralLearningAlgorithm;
/// use reCTBN::structure_learning::hypothesis_test::{F, ChiSquare};
/// use reCTBN::structure_learning::constraint_based_algorithm::CTPC;
/// #
/// # // Create the domain for a discrete node
/// # let mut domain = BTreeSet::new();
/// # domain.insert(String::from("A"));
/// # domain.insert(String::from("B"));
/// # domain.insert(String::from("C"));
/// # // Create the parameters for a discrete node using the domain
/// # let param = params::DiscreteStatesContinousTimeParams::new("n1".to_string(), domain);
/// # //Create the node n1 using the parameters
/// # let n1 = params::Params::DiscreteStatesContinousTime(param);
/// #
/// # let mut domain = BTreeSet::new();
/// # domain.insert(String::from("D"));
/// # domain.insert(String::from("E"));
/// # domain.insert(String::from("F"));
/// # let param = params::DiscreteStatesContinousTimeParams::new("n2".to_string(), domain);
/// # let n2 = params::Params::DiscreteStatesContinousTime(param);
/// #
/// # let mut domain = BTreeSet::new();
/// # domain.insert(String::from("G"));
/// # domain.insert(String::from("H"));
/// # domain.insert(String::from("I"));
/// # domain.insert(String::from("F"));
/// # let param = params::DiscreteStatesContinousTimeParams::new("n3".to_string(), domain);
/// # let n3 = params::Params::DiscreteStatesContinousTime(param);
/// #
/// # // Initialize a ctbn
/// # let mut net = CtbnNetwork::new();
/// #
/// # // Add the nodes and their edges
/// # let n1 = net.add_node(n1).unwrap();
/// # let n2 = net.add_node(n2).unwrap();
/// # let n3 = net.add_node(n3).unwrap();
/// # net.add_edge(n1, n2);
/// # net.add_edge(n1, n3);
/// # net.add_edge(n2, n3);
/// #
/// # match &mut net.get_node_mut(n1) {
/// # params::Params::DiscreteStatesContinousTime(param) => {
/// #     assert_eq!(
/// #         Ok(()),
/// #         param.set_cim(arr3(&[
/// #             [
/// #                 [-3.0, 2.0, 1.0],
/// #                 [1.5, -2.0, 0.5],
/// #                 [0.4, 0.6, -1.0]
/// #             ],
/// #         ]))
/// #     );
/// # }
/// # }
/// #
/// # match &mut net.get_node_mut(n2) {
/// # params::Params::DiscreteStatesContinousTime(param) => {
/// #     assert_eq!(
/// #         Ok(()),
/// #         param.set_cim(arr3(&[
/// #             [
/// #                 [-1.0, 0.5, 0.5],
/// #                 [3.0, -4.0, 1.0],
/// #                 [0.9, 0.1, -1.0]
/// #             ],
/// #             [
/// #                 [-6.0, 2.0, 4.0],
/// #                 [1.5, -2.0, 0.5],
/// #                 [3.0, 1.0, -4.0]
/// #             ],
/// #             [
/// #                 [-1.0, 0.1, 0.9],
/// #                 [2.0, -2.5, 0.5],
/// #                 [0.9, 0.1, -1.0]
/// #             ],
/// #         ]))
/// #     );
/// # }
/// # }
/// #
/// # match &mut net.get_node_mut(n3) {
/// # params::Params::DiscreteStatesContinousTime(param) => {
/// #     assert_eq!(
/// #         Ok(()),
/// #         param.set_cim(arr3(&[
/// #             [
/// #                 [-1.0, 0.5, 0.3, 0.2],
/// #                 [0.5, -4.0, 2.5, 1.0],
/// #                 [2.5, 0.5, -4.0, 1.0],
/// #                 [0.7, 0.2, 0.1, -1.0]
/// #             ],
/// #             [
/// #                 [-6.0, 2.0, 3.0, 1.0],
/// #                 [1.5, -3.0, 0.5, 1.0],
/// #                 [2.0, 1.3, -5.0, 1.7],
/// #                 [2.5, 0.5, 1.0, -4.0]
/// #             ],
/// #             [
/// #                 [-1.3, 0.3, 0.1, 0.9],
/// #                 [1.4, -4.0, 0.5, 2.1],
/// #                 [1.0, 1.5, -3.0, 0.5],
/// #                 [0.4, 0.3, 0.1, -0.8]
/// #             ],
/// #             [
/// #                 [-2.0, 1.0, 0.7, 0.3],
/// #                 [1.3, -5.9, 2.7, 1.9],
/// #                 [2.0, 1.5, -4.0, 0.5],
/// #                 [0.2, 0.7, 0.1, -1.0]
/// #             ],
/// #             [
/// #                 [-6.0, 1.0, 2.0, 3.0],
/// #                 [0.5, -3.0, 1.0, 1.5],
/// #                 [1.4, 2.1, -4.3, 0.8],
/// #                 [0.5, 1.0, 2.5, -4.0]
/// #             ],
/// #             [
/// #                 [-1.3, 0.9, 0.3, 0.1],
/// #                 [0.1, -1.3, 0.2, 1.0],
/// #                 [0.5, 1.0, -3.0, 1.5],
/// #                 [0.1, 0.4, 0.3, -0.8]
/// #             ],
/// #             [
/// #                 [-2.0, 1.0, 0.6, 0.4],
/// #                 [2.6, -7.1, 1.4, 3.1],
/// #                 [5.0, 1.0, -8.0, 2.0],
/// #                 [1.4, 0.4, 0.2, -2.0]
/// #             ],
/// #             [
/// #                 [-3.0, 1.0, 1.5, 0.5],
/// #                 [3.0, -6.0, 1.0, 2.0],
/// #                 [0.3, 0.5, -1.9, 1.1],
/// #                 [5.0, 1.0, 2.0, -8.0]
/// #             ],
/// #             [
/// #                 [-2.6, 0.6, 0.2, 1.8],
/// #                 [2.0, -6.0, 3.0, 1.0],
/// #                 [0.1, 0.5, -1.3, 0.7],
/// #                 [0.8, 0.6, 0.2, -1.6]
/// #             ],
/// #         ]))
/// #     );
/// # }
/// # }
/// #
/// # // Generate the trajectory
/// # let data = trajectory_generator(&net, 300, 30.0, Some(4164901764658873));
///
/// // Initialize the hypothesis tests to pass to the CTPC with their
/// // respective significance level `alpha`
/// let f = F::new(1e-6);
/// let chi_sq = ChiSquare::new(1e-4);
/// // Use the bayesian approach to learn the parameters
/// let parameter_learning = BayesianApproach { alpha: 1, tau:1.0 };
///
/// //Initialize CTPC
/// let ctpc = CTPC::new(parameter_learning, f, chi_sq);
///
/// // Learn the structure of the network from the generated trajectory
/// let net = ctpc.fit_transform(net, &data);
/// #
/// # // Compare the generated network with the original one
/// # assert_eq!(BTreeSet::new(), net.get_parent_set(0));
/// # assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
/// # assert_eq!(BTreeSet::from_iter(vec![0, 1]), net.get_parent_set(2));
/// ```
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

impl<P: ParameterLearning> StructuralLearningAlgorithm for CTPC<P> {
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
