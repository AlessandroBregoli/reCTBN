use std::collections::{HashMap, BTreeSet};
use petgraph::prelude::*;

use crate::node;
use crate::params;
use crate::network;




pub struct CtbnNetwork {
    network: petgraph::stable_graph::StableGraph<node::Node, ()>,
}

impl network::Network for CtbnNetwork {
    fn add_node(&mut self, mut n:  node::Node) -> Result<petgraph::graph::NodeIndex, network::NetworkError> {
        n.reset_params();
        Ok(self.network.add_node(n))        
    }

    fn add_edge(&mut self, parent: &petgraph::stable_graph::NodeIndex, child: &petgraph::graph::NodeIndex) {
        self.network.add_edge(parent.clone(), child.clone(), {});
        let mut p = self.network.node_weight_mut(child.clone()).unwrap();
        p.reset_params();
    }

    fn get_node_indices(&self) -> petgraph::stable_graph::NodeIndices<node::Node>{
        self.network.node_indices() 
    }

    fn get_node_weight(&self, node_idx: &petgraph::stable_graph::NodeIndex) -> &node::Node{
        self.network.node_weight(node_idx.clone()).unwrap()
    }

}
