use thiserror::Error;
use crate::params;
use ndarray::prelude::*;
use crate::node;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Error during node insertion")]
    NodeInsertionError(String)
}

pub trait Network {
    fn initialize_adj_matrix(&mut self);
    fn add_node(&mut self, n:  node::Node) -> Result<usize, NetworkError>;
    fn add_edge(&mut self, parent: usize, child: usize);
    fn get_node_indices(&self) -> std::ops::Range<usize>;
    fn get_node(&self, node_idx: usize) -> &node::Node;
    fn get_param_index_network(&self, node: usize, current_state: &Vec<params::StateType>) -> usize;
    fn get_parent_set(&self, node: usize) -> Vec<usize>;
    fn get_children_set(&self, node: usize) -> Vec<usize>;
}
