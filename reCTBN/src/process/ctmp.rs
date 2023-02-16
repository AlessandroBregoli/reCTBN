use std::collections::BTreeSet;

use ndarray::Array2;

use crate::{
    params::{Params, StateType},
    process,
};

use super::{NetworkProcess, NetworkProcessState};

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
            Some(_) => Err(process::NetworkError::NodeInsertionError(
                "CtmpProcess has only one node".to_string(),
            )),
        }
    }

    fn add_edge(&mut self, _parent: usize, _child: usize) {
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
    fn get_adj_matrix(&self) -> Option<Array2<u16>> {
        unimplemented!("CtmpProcess has only one node")
    }
}
