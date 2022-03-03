use crate::params::*;
use crate::network;
use crate::tools;
use ndarray::prelude::*;
use std::collections::BTreeSet;

pub fn MLE(net: Box<dyn network::Network>, 
           dataset: &tools::Dataset, 
           node: usize, 
           parent_set: Option<BTreeSet<usize>>) {
    
    let parent_set = match parent_set {
        Some(p) => p,
        None => net.get_parent_set(node)
    };
    

    

    
}
