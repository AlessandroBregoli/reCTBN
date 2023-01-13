//! Module containing score based algorithms like Hill Climbing and Tabu Search.

use std::collections::BTreeSet;

use crate::structure_learning::score_function::ScoreFunction;
use crate::structure_learning::StructureLearningAlgorithm;
use crate::{process, tools::Dataset};

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

impl<S: ScoreFunction> StructureLearningAlgorithm for HillClimbing<S> {
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
        //Iterate over each node to learn their parent set.
        for node in net.get_node_indices() {
            //Initialize an empty parent set.
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
            //Apply the learned parent_set to the network struct.
            parent_set.iter().for_each(|p| net.add_edge(*p, node));
        }

        return net;
    }
}
