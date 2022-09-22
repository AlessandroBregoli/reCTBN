use std::collections::BTreeSet;

use ndarray::prelude::*;

use crate::network;
use crate::params::{Params, ParamsTrait, StateType};

///CTBN network. It represents both the structure and the parameters of a CTBN. CtbnNetwork is
///composed by the following elements:
///- **adj_metrix**: a 2d ndarray representing the adjacency matrix
///- **nodes**: a vector containing all the nodes and their parameters.
///The index of a node inside the vector is also used as index for the adj_matrix.
///
///# Examples
///
///```
///    
/// use std::collections::BTreeSet;
/// use reCTBN::network::Network;
/// use reCTBN::params;
/// use reCTBN::ctbn::*;
///
/// //Create the domain for a discrete node
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
///
/// //Create the parameters for a discrete node using the domain
/// let param = params::DiscreteStatesContinousTimeParams::new("X1".to_string(), domain);
///
/// //Create the node using the parameters
/// let X1 = params::Params::DiscreteStatesContinousTime(param);
///
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
/// let param = params::DiscreteStatesContinousTimeParams::new("X2".to_string(), domain);
/// let X2 = params::Params::DiscreteStatesContinousTime(param);
///
/// //Initialize a ctbn
/// let mut net = CtbnNetwork::new();
///
/// //Add nodes
/// let X1 = net.add_node(X1).unwrap();
/// let X2 = net.add_node(X2).unwrap();
///
/// //Add an edge
/// net.add_edge(X1, X2);
///
/// //Get all the children of node X1
/// let cs = net.get_children_set(X1);
/// assert_eq!(&X2, cs.iter().next().unwrap());
/// ```
pub struct CtbnNetwork {
    adj_matrix: Option<Array2<u16>>,
    nodes: Vec<Params>,
}

impl CtbnNetwork {
    pub fn new() -> CtbnNetwork {
        CtbnNetwork {
            adj_matrix: None,
            nodes: Vec::new(),
        }
    }
}

impl network::Network for CtbnNetwork {
    fn initialize_adj_matrix(&mut self) {
        self.adj_matrix = Some(Array2::<u16>::zeros(
            (self.nodes.len(), self.nodes.len()).f(),
        ));
    }

    fn add_node(&mut self, mut n: Params) -> Result<usize, network::NetworkError> {
        n.reset_params();
        self.adj_matrix = Option::None;
        self.nodes.push(n);
        Ok(self.nodes.len() - 1)
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

    fn get_node_indices(&self) -> std::ops::Range<usize> {
        0..self.nodes.len()
    }

    fn get_number_of_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn get_node(&self, node_idx: usize) -> &Params {
        &self.nodes[node_idx]
    }

    fn get_node_mut(&mut self, node_idx: usize) -> &mut Params {
        &mut self.nodes[node_idx]
    }

    fn get_param_index_network(&self, node: usize, current_state: &Vec<StateType>) -> usize {
        self.adj_matrix
            .as_ref()
            .unwrap()
            .column(node)
            .iter()
            .enumerate()
            .fold((0, 1), |mut acc, x| {
                if x.1 > &0 {
                    acc.0 += self.nodes[x.0].state_to_index(&current_state[x.0]) * acc.1;
                    acc.1 *= self.nodes[x.0].get_reserved_space_as_parent();
                }
                acc
            })
            .0
    }

    fn get_param_index_from_custom_parent_set(
        &self,
        current_state: &Vec<StateType>,
        parent_set: &BTreeSet<usize>,
    ) -> usize {
        parent_set
            .iter()
            .fold((0, 1), |mut acc, x| {
                acc.0 += self.nodes[*x].state_to_index(&current_state[*x]) * acc.1;
                acc.1 *= self.nodes[*x].get_reserved_space_as_parent();
                acc
            })
            .0
    }

    fn get_parent_set(&self, node: usize) -> BTreeSet<usize> {
        self.adj_matrix
            .as_ref()
            .unwrap()
            .column(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| if x > &0 { Some(idx) } else { None })
            .collect()
    }

    fn get_children_set(&self, node: usize) -> BTreeSet<usize> {
        self.adj_matrix
            .as_ref()
            .unwrap()
            .row(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| if x > &0 { Some(idx) } else { None })
            .collect()
    }
}
