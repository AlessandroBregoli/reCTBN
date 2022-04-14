use crate::structure_learning::score_function::ScoreFunction;
use crate::structure_learning::StructureLearningAlgorithm;
use crate::tools;
use crate::network;
use std::collections::BTreeSet;

pub struct HillClimbing<S: ScoreFunction> {
    score_function: S,
    max_parent_set: Option<usize>
}

impl<S: ScoreFunction> HillClimbing<S> {
    pub fn init(score_function: S, max_parent_set: Option<usize>) -> HillClimbing<S> {
        HillClimbing { score_function, max_parent_set }
    }
}

impl<S: ScoreFunction> StructureLearningAlgorithm for HillClimbing<S> {
    fn fit_transform<T>(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network,
    {
        if net.get_number_of_nodes() != dataset.get_trajectories()[0].get_events().shape()[1] {
            panic!("Dataset and Network must have the same number of variables.")
        }

        let mut net = net;
        let max_parent_set = self.max_parent_set.unwrap_or(net.get_number_of_nodes());
        net.initialize_adj_matrix();
        for node in net.get_node_indices() {
            let mut parent_set: BTreeSet<usize> = BTreeSet::new();
            let mut current_ll = self.score_function.call(&net, node, &parent_set, dataset);
            let mut old_ll = f64::NEG_INFINITY;
            while current_ll > old_ll {
                old_ll = current_ll;
                for parent in net.get_node_indices() {
                    if parent == node {
                        continue;
                    }
                    let is_removed = parent_set.remove(&parent);
                    if !is_removed && parent_set.len() < max_parent_set {
                        parent_set.insert(parent);
                    }

                    let tmp_ll = self.score_function.call(&net, node, &parent_set, dataset);

                    if tmp_ll < current_ll {
                        if is_removed {
                            parent_set.insert(parent);
                        } else {
                            parent_set.remove(&parent);
                        }
                    } else {
                        current_ll = tmp_ll;
                    }
                }
            }
            parent_set.iter().for_each(|p| net.add_edge(*p, node));
        }

        return net;
    }
}
