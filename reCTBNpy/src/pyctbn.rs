use std::collections::BTreeSet;

use pyo3::prelude::*;
use reCTBN::{ctbn, network::Network};

#[pyclass]
pub struct PyCtbnNetwork {
    ctbn_network: ctbn::CtbnNetwork,
}

#[pymethods]
impl PyCtbnNetwork {
    #[new]
    pub fn new() -> Self {
        PyCtbnNetwork {
            ctbn_network: ctbn::CtbnNetwork::new(),
        }
    }

    pub fn get_number_of_nodes(&self) -> usize {
        self.ctbn_network.get_number_of_nodes()
    }

    pub fn add_edge(&mut self, parent: usize, child: usize) {
        self.ctbn_network.add_edge(parent, child);
    }

    pub fn get_parent_set(&self, node: usize) -> BTreeSet<usize> {
        self.ctbn_network.get_parent_set(node)
    }

    pub fn get_children_set(&self, node: usize) -> BTreeSet<usize> {
        self.ctbn_network.get_children_set(node)
    }
}
