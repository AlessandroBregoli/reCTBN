mod utils;
use std::collections::BTreeSet;
use std::f64::EPSILON;

use approx::AbsDiffEq;
use ndarray::arr3;
use reCTBN::process::{ctbn::*, ctmp::*};
use reCTBN::process::NetworkProcess;
use reCTBN::params::{self, ParamsTrait};
use utils::generate_discrete_time_continous_node;

#[test]
fn define_simpe_ctbn() {
    let _ = CtbnNetwork::new();
    assert!(true);
}

#[test]
fn add_node_to_ctbn() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    assert_eq!(&String::from("n1"), net.get_node(n1).get_label());
}

#[test]
fn add_edge_to_ctbn() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();
    net.add_edge(n1, n2);
    let cs = net.get_children_set(n1);
    assert_eq!(&n2, cs.iter().next().unwrap());
}

#[test]
fn children_and_parents() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();
    net.add_edge(n1, n2);
    let cs = net.get_children_set(n1);
    assert_eq!(&n2, cs.iter().next().unwrap());
    let ps = net.get_parent_set(n2);
    assert_eq!(&n1, ps.iter().next().unwrap());
}

#[test]
fn compute_index_ctbn() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();
    let n3 = net
        .add_node(generate_discrete_time_continous_node(String::from("n3"), 2))
        .unwrap();
    net.add_edge(n1, n2);
    net.add_edge(n3, n2);
    let idx = net.get_param_index_network(
        n2,
        &vec![
            params::StateType::Discrete(1),
            params::StateType::Discrete(1),
            params::StateType::Discrete(1),
        ],
    );
    assert_eq!(3, idx);

    let idx = net.get_param_index_network(
        n2,
        &vec![
            params::StateType::Discrete(0),
            params::StateType::Discrete(1),
            params::StateType::Discrete(1),
        ],
    );
    assert_eq!(2, idx);

    let idx = net.get_param_index_network(
        n2,
        &vec![
            params::StateType::Discrete(1),
            params::StateType::Discrete(1),
            params::StateType::Discrete(0),
        ],
    );
    assert_eq!(1, idx);
}

#[test]
fn compute_index_from_custom_parent_set() {
    let mut net = CtbnNetwork::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let _n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();
    let _n3 = net
        .add_node(generate_discrete_time_continous_node(String::from("n3"), 2))
        .unwrap();

    let idx = net.get_param_index_from_custom_parent_set(
        &vec![
            params::StateType::Discrete(0),
            params::StateType::Discrete(0),
            params::StateType::Discrete(1),
        ],
        &BTreeSet::from([1]),
    );
    assert_eq!(0, idx);

    let idx = net.get_param_index_from_custom_parent_set(
        &vec![
            params::StateType::Discrete(0),
            params::StateType::Discrete(0),
            params::StateType::Discrete(1),
        ],
        &BTreeSet::from([1, 2]),
    );
    assert_eq!(2, idx);
}

#[test]
fn simple_amalgamation() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    net.initialize_adj_matrix();

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]])));
        }
    }

    let ctmp = net.amalgamation();
    let p_ctbn = if let params::Params::DiscreteStatesContinousTime(p) = &net.get_node(0){
        p.get_cim().as_ref().unwrap()
    } else {
        unreachable!();
    };
    let p_ctmp = if let params::Params::DiscreteStatesContinousTime(p) = &ctmp.get_node(0) {
        p.get_cim().as_ref().unwrap()
    } else {
        unreachable!();
    };


    assert!(p_ctmp.abs_diff_eq(p_ctbn, std::f64::EPSILON));
}
