use std::collections::BTreeSet;

use crate::{
    params::{Params, StateType},
    process,
};

use super::{NetworkProcess, NetworkProcessState};
use log::warn;

/// This structure represents a Continuous Time Markov process
///
/// * Arguments
///
/// * `param` - An Option containing the parameters of the process
///
///```rust
/// use std::collections::BTreeSet;
/// use reCTBN::process::NetworkProcess;
/// use reCTBN::params;
/// use reCTBN::process::ctbn::*;
/// use ndarray::arr3;
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
/// match &mut net.get_node_mut(X1) {
///     params::Params::DiscreteStatesContinousTime(param) => {
///         assert_eq!(Ok(()), param.set_cim(arr3(&[[[-0.1, 0.1], [1.0, -1.0]]])));
///     }
/// }
///
/// match &mut net.get_node_mut(X2) {
///     params::Params::DiscreteStatesContinousTime(param) => {
///         assert_eq!(
///             Ok(()),
///             param.set_cim(arr3(&[
///                 [[-0.01, 0.01], [5.0, -5.0]],
///                 [[-5.0, 5.0], [0.01, -0.01]]
///             ]))
///         );
///     }
/// }
/// //Amalgamate the ctbn into a CtmpProcess
/// let ctmp = net.amalgamation();
///
/// //Extract the amalgamated params from the ctmp
///let params::Params::DiscreteStatesContinousTime(p_ctmp) = &ctmp.get_node(0);
///let p_ctmp = p_ctmp.get_cim().as_ref().unwrap();
///
/// //The shape of the params for an amalgamated ctmp can be computed as a Cartesian product of the
/// //domains variables of the ctbn
/// assert_eq!(p_ctmp.shape()[1], 4);
///```

pub struct CtmpProcess {
    param: Option<Params>,
}

impl CtmpProcess {
    pub fn new() -> CtmpProcess {
        CtmpProcess { param: None }
    }
}

impl NetworkProcess for CtmpProcess {
    fn initialize_adj_matrix(&mut self) {
        unimplemented!("CtmpProcess has only one node")
    }

    fn add_node(&mut self, n: crate::params::Params) -> Result<usize, process::NetworkError> {
        match self.param {
            None => {
                self.param = Some(n);
                Ok(0)
            }
            Some(_) => {
                warn!("A CTMP do not support more than one node");
                Err(process::NetworkError::NodeInsertionError(
                    "CtmpProcess has only one node".to_string(),
                ))
            }
        }
    }

    fn add_edge(&mut self, _parent: usize, _child: usize) {
        warn!("A CTMP cannot have edges");
        unimplemented!("CtmpProcess has only one node")
    }

    fn get_node_indices(&self) -> std::ops::Range<usize> {
        match self.param {
            None => 0..0,
            Some(_) => 0..1,
        }
    }

    fn get_number_of_nodes(&self) -> usize {
        match self.param {
            None => 0,
            Some(_) => 1,
        }
    }

    fn get_node(&self, node_idx: usize) -> &crate::params::Params {
        if node_idx == 0 {
            self.param.as_ref().unwrap()
        } else {
            unimplemented!("CtmpProcess has only one node")
        }
    }

    fn get_node_mut(&mut self, node_idx: usize) -> &mut crate::params::Params {
        if node_idx == 0 {
            self.param.as_mut().unwrap()
        } else {
            unimplemented!("CtmpProcess has only one node")
        }
    }

    fn get_param_index_network(&self, node: usize, current_state: &NetworkProcessState) -> usize {
        if node == 0 {
            match current_state[0] {
                StateType::Discrete(x) => x,
            }
        } else {
            unimplemented!("CtmpProcess has only one node")
        }
    }

    fn get_param_index_from_custom_parent_set(
        &self,
        _current_state: &NetworkProcessState,
        _parent_set: &BTreeSet<usize>,
    ) -> usize {
        unimplemented!("CtmpProcess has only one node")
    }

    fn get_parent_set(&self, node: usize) -> std::collections::BTreeSet<usize> {
        match self.param {
            Some(_) => {
                if node == 0 {
                    BTreeSet::new()
                } else {
                    unimplemented!("CtmpProcess has only one node")
                }
            }
            None => panic!("Uninitialized CtmpProcess"),
        }
    }

    fn get_children_set(&self, node: usize) -> std::collections::BTreeSet<usize> {
        match self.param {
            Some(_) => {
                if node == 0 {
                    BTreeSet::new()
                } else {
                    unimplemented!("CtmpProcess has only one node")
                }
            }
            None => panic!("Uninitialized CtmpProcess"),
        }
    }
}
