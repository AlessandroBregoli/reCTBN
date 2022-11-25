//! Defines methods for dealing with Probabilistic Graphical Models like the CTBNs

pub mod ctbn;
pub mod ctmp;

use std::collections::BTreeSet;

use thiserror::Error;

use crate::params;

/// Error types for trait Network
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Error during node insertion")]
    NodeInsertionError(String),
}

/// This type is used to represent a specific realization of a generic NetworkProcess
pub type NetworkProcessState = Vec<params::StateType>;

/// It defines the required methods for a structure used as a Probabilistic Graphical Models (such
/// as a CTBN).
pub trait NetworkProcess {
    fn initialize_adj_matrix(&mut self);
    fn add_node(&mut self, n: params::Params) -> Result<usize, NetworkError>;
    /// Add an **directed edge** between a two nodes of the network.
    ///
    /// # Arguments
    ///
    /// * `parent` - parent node.
    /// * `child` - child node.
    fn add_edge(&mut self, parent: usize, child: usize);

    /// Get all the indices of the nodes contained inside the network.
    fn get_node_indices(&self) -> std::ops::Range<usize>;

    /// Get the numbers of nodes contained in the network.
    fn get_number_of_nodes(&self) -> usize;

    /// Get the **node param**.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - node index value.
    ///
    /// # Return
    ///
    /// * The selected **node param**.
    fn get_node(&self, node_idx: usize) -> &params::Params;

    /// Get the **node param**.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - node index value.
    ///
    /// # Return
    ///
    /// * The selected **node mutable param**.
    fn get_node_mut(&mut self, node_idx: usize) -> &mut params::Params;

    /// Compute the index that must be used to access the parameters of a `node`, given a specific
    /// configuration of the network.
    ///
    /// Usually, the only values really used in `current_state` are the ones in the parent set of
    /// the `node`.
    ///
    /// # Arguments
    ///
    /// * `node` - selected node.
    /// * `current_state` - current configuration of the network.
    ///
    /// # Return
    ///
    /// * Index of the `node` relative to the network.
    fn get_param_index_network(&self, node: usize, current_state: &NetworkProcessState) -> usize;

    /// Compute the index that must be used to access the parameters of a `node`, given a specific
    /// configuration of the network and a generic `parent_set`.
    ///
    /// Usually, the only values really used in `current_state` are the ones in the parent set of
    /// the `node`.
    ///
    /// # Arguments
    ///
    /// * `current_state` - current configuration of the network.
    /// * `parent_set` - parent set of the selected `node`.
    ///
    /// # Return
    ///
    /// * Index of the `node` relative to the network.
    fn get_param_index_from_custom_parent_set(
        &self,
        current_state: &Vec<params::StateType>,
        parent_set: &BTreeSet<usize>,
    ) -> usize;

    /// Get the **parent set** of a given **node**.
    ///
    /// # Arguments
    ///
    /// * `node` - node index value.
    ///
    /// # Return
    ///
    /// * The **parent set** of the selected node.
    fn get_parent_set(&self, node: usize) -> BTreeSet<usize>;

    /// Get the **children set** of a given **node**.
    ///
    /// # Arguments
    ///
    /// * `node` - node index value.
    ///
    /// # Return
    ///
    /// * The **children set** of the selected node.
    fn get_children_set(&self, node: usize) -> BTreeSet<usize>;
}
