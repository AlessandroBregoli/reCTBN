mod utils;

use std::collections::BTreeSet;

use reCTBN::{
    params,
    params::ParamsTrait,
    process::{ctmp::*, NetworkProcess},
};
use utils::generate_discrete_time_continous_node;

#[test]
fn define_simple_ctmp() {
    let _ = CtmpProcess::new();
    assert!(true);
}

#[test]
fn add_node_to_ctmp() {
    let mut net = CtmpProcess::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    assert_eq!(&String::from("n1"), net.get_node(n1).get_label());
}

#[test]
fn add_two_nodes_to_ctmp() {
    let mut net = CtmpProcess::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let n2 = net.add_node(generate_discrete_time_continous_node(String::from("n1"), 2));

    match n2 {
        Ok(_) => assert!(false),
        Err(_) => assert!(true),
    };
}

#[test]
#[should_panic]
fn add_edge_to_ctmp() {
    let mut net = CtmpProcess::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let _n2 = net.add_node(generate_discrete_time_continous_node(String::from("n1"), 2));

    net.add_edge(0, 1)
}

#[test]
fn childen_and_parents() {
    let mut net = CtmpProcess::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    assert_eq!(0, net.get_parent_set(0).len());
    assert_eq!(0, net.get_children_set(0).len());
}

#[test]
#[should_panic]
fn get_childen_panic() {
    let net = CtmpProcess::new();
    net.get_children_set(0);
}

#[test]
#[should_panic]
fn get_childen_panic2() {
    let mut net = CtmpProcess::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    net.get_children_set(1);
}

#[test]
#[should_panic]
fn get_parent_panic() {
    let net = CtmpProcess::new();
    net.get_parent_set(0);
}

#[test]
#[should_panic]
fn get_parent_panic2() {
    let mut net = CtmpProcess::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    net.get_parent_set(1);
}

#[test]
fn compute_index_ctmp() {
    let mut net = CtmpProcess::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(
            String::from("n1"),
            10,
        ))
        .unwrap();

    let idx = net.get_param_index_network(n1, &vec![params::StateType::Discrete(6)]);
    assert_eq!(6, idx);
}

#[test]
#[should_panic]
fn compute_index_from_custom_parent_set_ctmp() {
    let mut net = CtmpProcess::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(
            String::from("n1"),
            10,
        ))
        .unwrap();

    let _idx = net.get_param_index_from_custom_parent_set(
        &vec![params::StateType::Discrete(6)],
        &BTreeSet::from([0])
    );
}
