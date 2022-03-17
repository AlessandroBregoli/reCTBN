mod utils;
use utils::*;

use rustyCTBN::parameter_learning::*;
use rustyCTBN::ctbn::*;
use rustyCTBN::network::Network;
use rustyCTBN::node;
use rustyCTBN::params;
use rustyCTBN::tools::*;
use ndarray::arr3;
use std::collections::BTreeSet;


#[macro_use]
extern crate approx;


#[test]
fn learn_binary_cim_MLE() {
    let mut net = CtbnNetwork::init();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"),2))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"),2))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.cim = Some(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]]));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.cim = Some(arr3(&[
                [[-1.0, 1.0], [4.0, -4.0]],
                [[-6.0, 6.0], [2.0, -2.0]],
            ]));
        }
    }

    let data = trajectory_generator(Box::new(&net), 100, 100.0);
    let mle = MLE{};
    let (CIM, M, T) = mle.fit(Box::new(&net), &data, 1, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [2, 2, 2]);
    assert!(CIM.abs_diff_eq(&arr3(&[
                [[-1.0, 1.0], [4.0, -4.0]],
                [[-6.0, 6.0], [2.0, -2.0]],
            ]), 0.2));
}


#[test]
fn learn_ternary_cim_MLE() {
    let mut net = CtbnNetwork::init();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"),3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"),3))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.cim = Some(arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]]));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.cim = Some(arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ]));
        }
    }

    let data = trajectory_generator(Box::new(&net), 100, 200.0);
    let mle = MLE{};
    let (CIM, M, T) = mle.fit(Box::new(&net), &data, 1, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [3, 3, 3]);
    assert!(CIM.abs_diff_eq(&arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ]), 0.2));
}

#[test]
fn learn_ternary_cim_MLE_no_parents() {
    let mut net = CtbnNetwork::init();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"),3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"),3))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.cim = Some(arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]]));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.cim = Some(arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ]));
        }
    }

    let data = trajectory_generator(Box::new(&net), 100, 200.0);
    let mle = MLE{};
    let (CIM, M, T) = mle.fit(Box::new(&net), &data, 0, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [1, 3, 3]);
    assert!(CIM.abs_diff_eq(&arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]]), 0.2));
}
