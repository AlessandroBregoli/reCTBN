use std::collections::BTreeSet; use petgraph::prelude::*;
use crate::params::*;

pub enum NodeType {
    DiscreteStatesContinousTime(DiscreteStatesContinousTimeParams)
}

pub struct Node {
    pub params: NodeType,
    pub label: String
}

impl Node {
    pub fn reset_params(&mut self) {
        match &mut self.params {
            NodeType::DiscreteStatesContinousTime(params) => {params.reset_params();}
        }
    }

    pub fn get_params(&self) -> &NodeType {
        &self.params
    }
    
    pub fn get_reserved_space_as_parent(&self) -> usize {
        match &self.params {
            NodeType::DiscreteStatesContinousTime(params) => params.get_reserved_space_as_parent()
        }
    }

    pub fn state_to_index(&self,state: &StateType) -> usize{
        match &self.params {
            NodeType::DiscreteStatesContinousTime(params) => params.state_to_index(state)
        }
    }

}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool{
        self.label == other.label
    }
}


