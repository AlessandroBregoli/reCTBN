//! Module containing score based algorithms like Hill Climbing and Tabu Search.

use log::info;
use std::collections::BTreeSet;

use crate::structure_learning::score_function::ScoreFunction;
use crate::structure_learning::StructuralLearningAlgorithm;
use crate::{process, tools::Dataset};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::ParallelExtend;

/// HillClimbing functor
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
/// use reCTBN::structure_learning::StructuralLearningAlgorithm;
/// use reCTBN::structure_learning::score_based_algorithm::*;
/// use reCTBN::structure_learning::score_function::*;
/// use reCTBN::parameter_learning::Tau;
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
/// // Initialize the BIC score function
/// let bic = BIC::new(1, Tau::Constant(0.1));
///
/// //Initialize HC
/// let hc = HillClimbing::new(bic, None);
///
/// // Learn the structure of the network from the generated trajectory
/// let net = hc.fit_transform(net, &data);
/// #
/// # // Compare the generated network with the original one
/// # assert_eq!(BTreeSet::new(), net.get_parent_set(0));
/// # assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
/// # assert_eq!(BTreeSet::from_iter(vec![0, 1]), net.get_parent_set(2));
/// ````
pub struct HillClimbing<S: ScoreFunction> {
    score_function: S,
    max_parent_set: Option<usize>,
}

impl<S: ScoreFunction> HillClimbing<S> {
    pub fn new(score_function: S, max_parent_set: Option<usize>) -> HillClimbing<S> {
        HillClimbing {
            score_function,
            max_parent_set,
        }
    }
}

impl<S: ScoreFunction> StructuralLearningAlgorithm for HillClimbing<S> {
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
        //Check if the max_parent_set constraint is present.
        let max_parent_set = self.max_parent_set.unwrap_or(net.get_number_of_nodes());
        //Reset the adj matrix
        net.initialize_adj_matrix();
        let mut learned_parent_sets: Vec<(usize, BTreeSet<usize>)> = vec![];
        //Iterate over each node to learn their parent set.
        learned_parent_sets.par_extend(net.get_node_indices().into_par_iter().map(|node| {
            //Initialize an empty parent set.
            info!("Learning node {}", node);
            let mut parent_set: BTreeSet<usize> = BTreeSet::new();
            //Compute the score for the empty parent set
            let mut current_score = self.score_function.call(&net, node, &parent_set, dataset);
            //Set the old score to -\infty.
            let mut old_score = f64::NEG_INFINITY;
            //Iterate until convergence
            while current_score > old_score {
                //Save the current_score.
                old_score = current_score;
                //Iterate over each node.
                for parent in net.get_node_indices() {
                    //Continue if the parent and the node are the same.
                    if parent == node {
                        continue;
                    }
                    //Try to remove parent from the parent_set.
                    let is_removed = parent_set.remove(&parent);
                    //If parent was not in the parent_set add it.
                    if !is_removed && parent_set.len() < max_parent_set {
                        parent_set.insert(parent);
                    }
                    //Compute the score with the modified parent_set.
                    let tmp_score = self.score_function.call(&net, node, &parent_set, dataset);
                    //If tmp_score is worst than current_score revert the change to the parent set
                    if tmp_score < current_score {
                        if is_removed {
                            parent_set.insert(parent);
                        } else {
                            parent_set.remove(&parent);
                        }
                    }
                    //Otherwise save the computed score as current_score
                    else {
                        current_score = tmp_score;
                    }
                }
            }
            (node, parent_set)
        }));

        for (child_node, candidate_parent_set) in learned_parent_sets {
            for parent_node in candidate_parent_set.iter() {
                net.add_edge(*parent_node, child_node);
            }
        }
        return net;
    }
}
