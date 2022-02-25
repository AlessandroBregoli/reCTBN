use std::collections::{HashMap, BTreeSet};
use ndarray::prelude::*;
use crate::node;
use crate::params::StateType;
use crate::network;




///CTBN network. It represents both the structure and the parameters of a CTBN. CtbnNetwork is
///composed by the following elements:
///- **adj_metrix**: a 2d ndarray representing the adjacency matrix
///- **nodes**: a vector containing all the nodes and their parameters.
///The index of a node inside the vector is also used as index for the adj_matrix.
///
///# Examples
///
///```
///    
/// use std::collections::BTreeSet;
/// use rustyCTBN::network::Network;
/// use rustyCTBN::node;
/// use rustyCTBN::params;
/// use rustyCTBN::ctbn::*;
///
/// //Create the domain for a discrete node
/// let mut domain = BTreeSet::new(); 
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
///
/// //Create the parameters for a discrete node using the domain
/// let params = params::DiscreteStatesContinousTimeParams::init(domain); 
///
/// //Create the node using the parameters
/// let X1 = node::Node::init(node::NodeType::DiscreteStatesContinousTime(params),String::from("X1"));
///
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
/// let params = params::DiscreteStatesContinousTimeParams::init(domain);
/// let X2 = node::Node::init(node::NodeType::DiscreteStatesContinousTime(params),String::from("X2"));
/// 
/// //Initialize a ctbn
/// let mut net = CtbnNetwork::init();
///
/// //Add nodes
/// let X1 = net.add_node(X1).unwrap();
/// let X2 = net.add_node(X2).unwrap();
/// 
/// //Add an edge
/// net.add_edge(X1, X2);
///
/// //Get all the children of node X1
/// let cs = net.get_children_set(X1);
/// assert_eq!(X2, cs[0]);
/// ```
pub struct CtbnNetwork {
    adj_matrix: Option<Array2<u16>>,
    nodes: Vec<node::Node>
}


impl CtbnNetwork {
    pub fn init() -> CtbnNetwork {
        CtbnNetwork {
            adj_matrix: None,
            nodes: Vec::new()
        }
    }
}

impl network::Network for CtbnNetwork {
    fn initialize_adj_matrix(&mut self) {
        self.adj_matrix = Some(Array2::<u16>::zeros((self.nodes.len(), self.nodes.len()).f()));

    }

    fn add_node(&mut self, mut n:  node::Node) -> Result<usize, network::NetworkError> {
        n.reset_params();
        self.adj_matrix = Option::None;
        self.nodes.push(n);
        Ok(self.nodes.len() -1)        
    }

    fn add_edge(&mut self, parent: usize, child: usize) {
        if let None = self.adj_matrix {
            self.initialize_adj_matrix();
        }

        if let Some(network) = &mut self.adj_matrix {
            network[[parent, child]] = 1;
            self.nodes[child].reset_params();
        }
    }

    fn get_node_indices(&self) -> std::ops::Range<usize>{
        0..self.nodes.len()
    }

    fn get_node(&self, node_idx: usize) -> &node::Node{
        &self.nodes[node_idx]
    }


    fn get_param_index_network(&self, node: usize, current_state: &Vec<StateType>) -> usize{
        self.adj_matrix.as_ref().unwrap().column(node).iter().enumerate().fold((0, 1), |mut acc, x| {
            if x.1 > &0 {
                acc.0 += self.nodes[x.0].state_to_index(&current_state[x.0]) * acc.1;
                acc.1 *= self.nodes[x.0].get_reserved_space_as_parent();
            }
            acc
        }).0
    }

    fn get_parent_set(&self, node: usize) -> Vec<usize> {
        self.adj_matrix.as_ref()
            .unwrap()
            .column(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| {
                if x > &0 {
                    Some(idx)
                } else {
                    None
                }
            }).collect()
    }

    fn get_children_set(&self, node: usize) -> Vec<usize>{
        self.adj_matrix.as_ref()
            .unwrap()
            .row(node)
            .iter()
            .enumerate()
            .filter_map(|(idx, x)| {
                if x > &0 {
                    Some(idx)
                } else {
                    None
                }
            }).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use crate::node;
    use crate::params;
    use std::collections::BTreeSet;

    fn define_binary_node(name: String) -> node::Node {
        let mut domain = BTreeSet::new();
        domain.insert(String::from("A"));
        domain.insert(String::from("B"));
        let params = params::DiscreteStatesContinousTimeParams::init(domain);
        let n = node::Node::init(node::NodeType::DiscreteStatesContinousTime(params),name);
        return n;
    }

    #[test]
    fn define_simpe_ctbn() {
        let _ = CtbnNetwork::init();
        assert!(true);
    }

    #[test]
    fn add_node_to_ctbn() {
        let mut net = CtbnNetwork::init();
        let n1 = net.add_node(define_binary_node(String::from("n1"))).unwrap();
        assert_eq!(String::from("n1"), net.get_node(n1).label);
    }

    #[test]
    fn add_edge_to_ctbn() {
        let mut net = CtbnNetwork::init();
        let n1 = net.add_node(define_binary_node(String::from("n1"))).unwrap();
        let n2 = net.add_node(define_binary_node(String::from("n2"))).unwrap();
        net.add_edge(n1, n2);
        let cs = net.get_children_set(n1);
        assert_eq!(n2, cs[0]);
    }

    #[test]
    fn children_and_parents() {
        let mut net = CtbnNetwork::init();
        let n1 = net.add_node(define_binary_node(String::from("n1"))).unwrap();
        let n2 = net.add_node(define_binary_node(String::from("n2"))).unwrap();
        net.add_edge(n1, n2);
        let cs = net.get_children_set(n1);
        assert_eq!(n2, cs[0]);
        let ps = net.get_parent_set(n2);
        assert_eq!(n1, ps[0]);
    }


    #[test]
    fn compute_index_ctbn() {
        let mut net = CtbnNetwork::init();
        let n1 = net.add_node(define_binary_node(String::from("n1"))).unwrap();
        let n2 = net.add_node(define_binary_node(String::from("n2"))).unwrap();
        let n3 = net.add_node(define_binary_node(String::from("n3"))).unwrap();
        net.add_edge(n1, n2);
        net.add_edge(n3, n2);
        let idx = net.get_param_index_network(n2, &vec![
                                              params::StateType::Discrete(1), 
                                              params::StateType::Discrete(1), 
                                              params::StateType::Discrete(1)]);
        assert_eq!(3, idx);


        let idx = net.get_param_index_network(n2, &vec![
                                              params::StateType::Discrete(0), 
                                              params::StateType::Discrete(1), 
                                              params::StateType::Discrete(1)]);
        assert_eq!(2, idx);


        let idx = net.get_param_index_network(n2, &vec![
                                              params::StateType::Discrete(1), 
                                              params::StateType::Discrete(1), 
                                              params::StateType::Discrete(0)]);
        assert_eq!(1, idx);
    }
}
