//! Continuous Time Bayesian Network

use std::collections::BTreeSet;

use log::info;
use ndarray::prelude::*;

use crate::params::{DiscreteStatesContinousTimeParams, Params, ParamsTrait, StateType};
use crate::process;

use super::ctmp::CtmpProcess;
use super::{NetworkProcess, NetworkProcessState};

/// It represents both the structure and the parameters of a CTBN.
///
/// # Arguments
///
/// * `adj_matrix` - A 2D ndarray representing the adjacency matrix
/// * `nodes` - A vector containing all the nodes and their parameters.
///
/// The index of a node inside the vector is also used as index for the `adj_matrix`.
///
/// # Example
///
/// ```rust
/// use std::collections::BTreeSet;
/// use reCTBN::process::NetworkProcess;
/// use reCTBN::params;
/// use reCTBN::process::ctbn::*;
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

    ///Transform the **CTBN** into a **CTMP**
    ///
    /// # Return
    ///
    /// * The equivalent *CtmpProcess* computed from the current CtbnNetwork
    pub fn amalgamation(&self) -> CtmpProcess {
        info!("Network Amalgamation Started");

        // Get the domanin (cardinality) for each node in the network
        let variables_domain =
            Array1::from_iter(self.nodes.iter().map(|x| x.get_reserved_space_as_parent()));

        // The state space size for a ctmp generated from a ctbn is equal to the product of the
        // caridalities of each node in the ctbn.
        let state_space = variables_domain.product();
        let variables_set = BTreeSet::from_iter(self.get_node_indices());
        let mut amalgamated_cim: Array3<f64> = Array::zeros((1, state_space, state_space));

        for idx_current_state in 0..state_space {
            //Compute the state of the ctbn given the state of the ctmp
            let current_state = CtbnNetwork::idx_to_state(&variables_domain, idx_current_state);
            let current_state_statetype: NetworkProcessState = current_state
                .iter()
                .map(|x| StateType::Discrete(*x))
                .collect();

            // Amalgamation for the current state (Generation of one row of the `amalgamated_cim`)
            for idx_node in 0..self.nodes.len() {
                let p = match self.get_node(idx_node) {
                    Params::DiscreteStatesContinousTime(p) => p,
                };

                // Add the transition intensities for each possible configuration of the node
                // `idx_node` in the `amalgamated_cim`
                for next_node_state in 0..variables_domain[idx_node] {
                    let mut next_state = current_state.clone();
                    next_state[idx_node] = next_node_state;

                    let next_state_statetype: NetworkProcessState =
                        next_state.iter().map(|x| StateType::Discrete(*x)).collect();
                    let idx_next_state = self.get_param_index_from_custom_parent_set(
                        &next_state_statetype,
                        &variables_set,
                    );
                    amalgamated_cim[[0, idx_current_state, idx_next_state]] +=
                        p.get_cim().as_ref().unwrap()[[
                            self.get_param_index_network(idx_node, &current_state_statetype),
                            current_state[idx_node],
                            next_node_state,
                        ]];
                }
            }
        }

        let mut amalgamated_param = DiscreteStatesContinousTimeParams::new(
            "ctmp".to_string(),
            BTreeSet::from_iter((0..state_space).map(|x| x.to_string())),
        );

        amalgamated_param.set_cim(amalgamated_cim).unwrap();

        let mut ctmp = CtmpProcess::new();

        ctmp.add_node(Params::DiscreteStatesContinousTime(amalgamated_param))
            .unwrap();
        return ctmp;
    }

    /// Compute the state for each node given an index and a set of ordered variables
    ///
    /// # Arguments
    ///
    /// * `variables_domain` - domain of the considered variables
    /// * `state` - specific configuration of the nodes represented with a single number.
    ///
    /// # Return
    ///
    /// * An array containing the state of each node
    pub fn idx_to_state(variables_domain: &Array1<usize>, state: usize) -> Array1<usize> {
        let mut state = state;
        let mut array_state = Array1::zeros(variables_domain.shape()[0]);
        for (idx, var) in variables_domain.indexed_iter() {
            array_state[idx] = state % var;
            state = state / var;
        }

        return array_state;
    }
    /// Get the Adjacency Matrix.
    pub fn get_adj_matrix(&self) -> Option<&Array2<u16>> {
        self.adj_matrix.as_ref()
    }
}

impl process::NetworkProcess for CtbnNetwork {
    fn initialize_adj_matrix(&mut self) {
        self.adj_matrix = Some(Array2::<u16>::zeros(
            (self.nodes.len(), self.nodes.len()).f(),
        ));
    }

    fn add_node(&mut self, mut n: Params) -> Result<usize, process::NetworkError> {
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

    fn get_param_index_network(&self, node: usize, current_state: &NetworkProcessState) -> usize {
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
        current_state: &NetworkProcessState,
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
