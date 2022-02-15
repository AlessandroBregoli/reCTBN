use petgraph::prelude::*;
use crate::node;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Error during node insertion")]
    InsertionError(String)
}

pub trait Network {
    fn add_node(&mut self, n:  node::Node) -> Result<petgraph::graph::NodeIndex, NetworkError>;
}


