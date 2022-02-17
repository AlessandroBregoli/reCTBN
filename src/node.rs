use std::collections::BTreeSet;
use crate::params;


pub struct Node {
    pub params:  Box<dyn params::Params>,
    pub label: String
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool{
        self.label == other.label
    }
}


