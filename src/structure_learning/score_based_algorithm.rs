use crate::structure_learning::score_function::ScoreFunction;
use crate::structure_learning::StructureLearningAlgorithm;
use crate::tools;
use crate::network;
use std::collections::BTreeSet;

pub struct HillClimbing<S: ScoreFunction> {
    score_function: S,
}

impl<S: ScoreFunction> HillClimbing<S> {
    pub fn init(score_function: S) -> HillClimbing<S> {
        HillClimbing { score_function }
    }
}

impl<S: ScoreFunction> StructureLearningAlgorithm for HillClimbing<S> {
    fn fit_transform<T>(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network,
    {
        let mut net = net;
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
                    if !is_removed {
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
