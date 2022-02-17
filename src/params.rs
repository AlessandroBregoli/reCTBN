use ndarray::prelude::*;
use std::collections::{HashMap, BTreeSet};
use petgraph::prelude::*;



pub trait Params {

    fn add_parent(&mut self, p: &petgraph::stable_graph::NodeIndex);
}

pub struct DiscreteStatesContinousTimeParams {
    domain: BTreeSet<String>,
    parents: BTreeSet<petgraph::stable_graph::NodeIndex>,
    cim: Option<Array3<f64>>,
    transitions: Option<Array3<u64>>,
    residence_time: Option<Array2<f64>>
}

impl DiscreteStatesContinousTimeParams  {
    fn init(domain: BTreeSet<String>) -> DiscreteStatesContinousTimeParams {
        DiscreteStatesContinousTimeParams {
            domain: domain,
            parents: BTreeSet::new(),
            cim: Option::None,
            transitions: Option::None,
            residence_time: Option::None
        }
    }
}

impl Params for DiscreteStatesContinousTimeParams {
    fn add_parent(&mut self, p: &petgraph::stable_graph::NodeIndex) {
        self.parents.insert(p.clone());
        self.cim = Option::None;
        self.transitions = Option::None;
        self.residence_time = Option::None;
    }
}
