use petgraph::prelude::*;
use crate::node;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Error during node insertion")]
    NodeInsertionError(String)
}

pub trait Network {
    fn add_node(&mut self, n:  node::Node) -> Result<petgraph::graph::NodeIndex, NetworkError>;
    fn add_edge(&mut self, parent: &petgraph::stable_graph::NodeIndex, child: &petgraph::graph::NodeIndex);
    fn get_node_indices(&self) -> petgraph::stable_graph::NodeIndices<node::Node>;
    fn get_node_weight(&self, node_idx: &petgraph::stable_graph::NodeIndex) -> &node::Node;
}
