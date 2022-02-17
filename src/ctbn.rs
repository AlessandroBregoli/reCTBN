use std::collections::{HashMap, BTreeSet};
use petgraph::prelude::*;

use crate::node;
use crate::params;
use crate::network;




pub struct CtbnNetwork {
    network: petgraph::stable_graph::StableGraph<node::Node, ()>,
}

impl network::Network for CtbnNetwork {
    fn add_node(&mut self, n:  node::Node) -> Result<petgraph::graph::NodeIndex, network::NetworkError> {
        match &n.params {
            node::ParamsType::DiscreteStatesContinousTime(_) => {
                if self.network.node_weights().any(|x| x.label == n.label) {
                    //TODO: Insert a better error description
                    return Err(network::NetworkError::NodeInsertionError(String::from("Label already used")));
                }
                let idx = self.network.add_node(n);
                Ok(idx)
            },
            //TODO: Insert a better error description
            _ => Err(network::NetworkError::NodeInsertionError(String::from("unsupported node")))
        }
    }

    fn add_edge(&mut self, parent: &petgraph::stable_graph::NodeIndex, child: &petgraph::graph::NodeIndex) {
        self.network.add_edge(parent.clone(), child.clone(), {});
        let mut p = self.network.node_weight(child.clone());
        match p.
    }

    fn get_node_indices(&self) -> petgraph::stable_graph::NodeIndices<node::Node>{
        self.network.node_indices() 
    }

    fn get_node_weight(&self, node_idx: &petgraph::stable_graph::NodeIndex) -> &node::Node{
        self.network.node_weight(node_idx.clone()).unwrap()
    }

}
