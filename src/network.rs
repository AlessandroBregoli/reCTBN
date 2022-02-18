use petgraph::prelude::*; use crate::node;
use thiserror::Error;
use crate::params;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Error during node insertion")]
    NodeInsertionError(String)
}

pub trait Network {
    fn add_node(&mut self, n:  node::Node) -> Result<petgraph::graph::NodeIndex, NetworkError>;
    fn add_edge(&mut self, parent: &petgraph::stable_graph::NodeIndex, child: &petgraph::graph::NodeIndex);
    fn get_node_indices(&self) -> petgraph::stable_graph::NodeIndices<node::Node>;
    fn get_node(&self, node_idx: &petgraph::stable_graph::NodeIndex) -> &node::Node;
    fn get_param_index_parents(&self, node: &petgraph::stable_graph::NodeIndex, u: &Vec<params::StateType>) -> usize;
    fn get_param_index_network(&self, node: &petgraph::stable_graph::NodeIndex, current_state: &Vec<params::StateType>) -> usize;
}
