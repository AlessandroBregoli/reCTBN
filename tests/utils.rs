use rustyCTBN::params;
use rustyCTBN::node;
use std::collections::BTreeSet;

pub fn generate_discrete_time_continous_node(name: String, cardinality: usize) -> node::Node {
    node::Node::init(params::Params::DiscreteStatesContinousTime(generate_discrete_time_continous_param(cardinality)), name)
}


pub fn generate_discrete_time_continous_param(cardinality: usize) -> params::DiscreteStatesContinousTimeParams{
    let mut domain: BTreeSet<String> = (0..cardinality).map(|x| x.to_string()).collect();
    params::DiscreteStatesContinousTimeParams::init(domain)
}


