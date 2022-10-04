use std::collections::BTreeSet;

use thiserror::Error;

use crate::params;

/// Error types for trait Network
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Error during node insertion")]
    NodeInsertionError(String),
}

///Network
///The Network trait define the required methods for a structure used as pgm (such as ctbn).
pub trait Network {
    fn initialize_adj_matrix(&mut self);
    fn add_node(&mut self, n: params::Params) -> Result<usize, NetworkError>;
    fn add_edge(&mut self, parent: usize, child: usize);

    ///Get all the indices of the nodes contained inside the network
    fn get_node_indices(&self) -> std::ops::Range<usize>;
    fn get_number_of_nodes(&self) -> usize;
    fn get_node(&self, node_idx: usize) -> &params::Params;
    fn get_node_mut(&mut self, node_idx: usize) -> &mut params::Params;

    ///Compute the index that must be used to access the parameters of a node given a specific
    ///configuration of the network. Usually, the only values really used in *current_state* are
    ///the ones in the parent set of the *node*.
    fn get_param_index_network(&self, node: usize, current_state: &Vec<params::StateType>)
        -> usize;

    ///Compute the index that must be used to access the parameters of a node given a specific
    ///configuration of the network and a generic parent_set. Usually, the only values really used
    ///in *current_state* are the ones in the parent set of the *node*.
    fn get_param_index_from_custom_parent_set(
        &self,
        current_state: &Vec<params::StateType>,
        parent_set: &BTreeSet<usize>,
    ) -> usize;
    fn get_parent_set(&self, node: usize) -> BTreeSet<usize>;
    fn get_children_set(&self, node: usize) -> BTreeSet<usize>;
}