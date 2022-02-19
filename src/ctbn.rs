use std::collections::{HashMap, BTreeSet};
use ndarray::prelude::*;
use crate::node;
use crate::params::StateType;
use crate::network;




pub struct CtbnNetwork {
    adj_matrix: Option<Array2<u16>>,
    nodes: Vec<node::Node>
}

impl network::Network for CtbnNetwork {
    fn initialize_adj_matrix(&mut self) {
        self.adj_matrix = Some(Array2::<u16>::zeros((self.nodes.len(), self.nodes.len()).f()));

    }

    fn add_node(&mut self, mut n:  node::Node) -> Result<usize, network::NetworkError> {
        n.reset_params();
        self.adj_matrix = Option::None;
        self.nodes.push(n);
        Ok(self.nodes.len() -1)        
    }

    fn add_edge(&mut self, parent: usize, child: usize) {
        if let None = self.adj_matrix {
            self.initialize_adj_matrix();
        }

        if let Some(network) = &mut self.adj_matrix {
            network[[parent, child]] = 1;
            self.nodes[child].reset_params();
        }
    }

    fn get_node_indices(&self) -> std::ops::Range<usize>{
        0..self.nodes.len()
    }

    fn get_node(&self, node_idx: usize) -> &node::Node{
        &self.nodes[node_idx]
    }


    fn get_param_index_network(&self, node: usize, current_state: &Vec<StateType>) -> usize{
        self.adj_matrix.as_ref().unwrap().column(node).iter().enumerate().fold((0, 1), |mut acc, x| {
            if x.1 > &0 {
                acc.0 += self.nodes[x.0].state_to_index(&current_state[x.0]) * acc.1;
                acc.1 *= self.nodes[x.0].get_reserved_space_as_parent();
            }
            acc
        }).0
    }

    fn get_parent_set(&self, node: usize) -> Vec<usize> {
        self.adj_matrix.as_ref()
            .unwrap()
            .column(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| {
                if x > &0 {
                    Some(idx)
                } else {
                    None
                }
            }).collect()
    }
    fn get_children_set(&self, node: usize) -> Vec<usize>{
        self.adj_matrix.as_ref()
            .unwrap()
            .row(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| {
                if x > &0 {
                    Some(idx)
                } else {
                    None
                }
            }).collect()
    }

}
