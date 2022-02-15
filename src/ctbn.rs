use std::collections::HashMap;
use petgraph::prelude::*;

use crate::node;
use crate::params;
use crate::network;



pub struct CtbnParams {
    cim: Option<params::CIM>,
    transitions: Option<params::M>,
    residence_time: Option<params::T>
}

impl CtbnParams {
    fn init() -> CtbnParams {
        CtbnParams {
            cim: Option::None,
            transitions: Option::None,
            residence_time: Option::None
        }
    }
}

pub struct CtbnNetwork {
    network: petgraph::stable_graph::StableGraph<node::Node, u64>,
    params: HashMap<petgraph::graph::NodeIndex,CtbnParams>,
}

impl network::Network for CtbnNetwork {
    fn add_node(&mut self, n:  node::Node) -> Result<petgraph::graph::NodeIndex, network::NetworkError> {
        match &n.domain {
            node::DomainType::Discrete(_) => {
                let idx = self.network.add_node(n);
                self.params.insert(idx, CtbnParams::init());
                Ok(idx)
            },
            _ => Err(network::NetworkError::InsertionError(String::from("unsupported node")))
        }

    }
}


