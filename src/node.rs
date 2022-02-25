use crate::params::*;


pub struct Node {
    pub params: Box<dyn Params>,
    pub label: String
}

impl Node {
    pub fn init(params: Box<dyn Params>, label: String) -> Node {
        Node{
            params: params,
            label:label
        }
    }

}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool{
        self.label == other.label
    }
}


