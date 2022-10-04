use std::collections::BTreeSet;

use crate::{pyparams, pytools};
use pyo3::prelude::*;
use reCTBN::{ctbn, network::Network, params, tools, params::Params};

#[pyclass]
pub struct PyCtbnNetwork(pub ctbn::CtbnNetwork);

#[pymethods]
impl PyCtbnNetwork {
    #[new]
    pub fn new() -> Self {
        PyCtbnNetwork(ctbn::CtbnNetwork::new())
    }

    pub fn add_node(&mut self, n: pyparams::PyParams) {
        self.0.add_node(n.0);
    }

    pub fn get_number_of_nodes(&self) -> usize {
        self.0.get_number_of_nodes()
    }

    pub fn add_edge(&mut self, parent: usize, child: usize) {
        self.0.add_edge(parent, child);
    }

    pub fn get_node_indices(&self) -> BTreeSet<usize> {
        self.0.get_node_indices().collect()
    }

    pub fn get_parent_set(&self, node: usize) -> BTreeSet<usize> {
        self.0.get_parent_set(node)
    }

    pub fn get_children_set(&self, node: usize) -> BTreeSet<usize> {
        self.0.get_children_set(node)
    }

    pub fn set_node(&mut self, node_idx: usize, n: pyparams::PyParams) {
        match &n.0 {
            Params::DiscreteStatesContinousTime(new_p) => {
                if let Params::DiscreteStatesContinousTime(p) = self.0.get_node_mut(node_idx){
                    p.set_cim(new_p.get_cim().as_ref().unwrap().clone()).unwrap();

                }
                else {
                    panic!("Node type mismatch")
                }
            }
        }
    }

    pub fn trajectory_generator(
        &self,
        n_trajectories: u64,
        t_end: f64,
        seed: Option<u64>,
    ) -> pytools::PyDataset {
        pytools::PyDataset(tools::trajectory_generator(
            &self.0,
            n_trajectories,
            t_end,
            seed,
        ))
    }
}
