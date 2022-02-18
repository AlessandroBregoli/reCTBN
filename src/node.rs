use std::collections::BTreeSet;
use petgraph::prelude::*;
use crate::params::*;

pub enum NodeType {
    DiscreteStatesContinousTime(DiscreteStatesContinousTimeParams)
}

pub struct Node {
    pub params: NodeType,
    pub label: String
}

impl Node {
    pub fn add_parent(&mut self, parent: &petgraph::stable_graph::NodeIndex) {
        match  &mut self.params {
            NodeType::DiscreteStatesContinousTime(params) => {params.add_parent(parent);}
        }
    }

    pub fn reset_params(&mut self) {
        match &mut self.params {
            NodeType::DiscreteStatesContinousTime(params) => {params.reset_params();}
        }
    }

    pub fn get_params(&self) -> &NodeType {
        &self.params
    }

}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool{
        self.label == other.label
    }
}


