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


fn learn_binary_cim<T: ParameterLearning> (pl: T) {
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
            assert_eq!(Ok(()), param.set_cim(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]])));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[
                [[-1.0, 1.0], [4.0, -4.0]],
                [[-6.0, 6.0], [2.0, -2.0]],
            ])));
        }
    }

    let data = trajectory_generator(&net, 100, 100.0);
    let (CIM, M, T) = pl.fit(&net, &data, 1, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [2, 2, 2]);
    assert!(CIM.abs_diff_eq(&arr3(&[
                [[-1.0, 1.0], [4.0, -4.0]],
                [[-6.0, 6.0], [2.0, -2.0]],
            ]), 0.2));
}

#[test]
fn learn_binary_cim_MLE() {
    let mle = MLE{};
    learn_binary_cim(mle);
}


#[test]
fn learn_binary_cim_BA() {
    let ba = BayesianApproach{
        alpha: 1,
        tau: 1.0};
    learn_binary_cim(ba);
}

fn learn_ternary_cim<T: ParameterLearning> (pl: T) {
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
            assert_eq!(Ok(()), param.set_cim(arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]])));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ])));
        }
    }

    let data = trajectory_generator(&net, 100, 200.0);
    let (CIM, M, T) = pl.fit(&net, &data, 1, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [3, 3, 3]);
    assert!(CIM.abs_diff_eq(&arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ]), 0.2));
}


#[test]
fn learn_ternary_cim_MLE() {
    let mle = MLE{};
    learn_ternary_cim(mle);
}


#[test]
fn learn_ternary_cim_BA() {
    let ba = BayesianApproach{
        alpha: 1,
        tau: 1.0};
    learn_ternary_cim(ba);
}

fn learn_ternary_cim_no_parents<T: ParameterLearning> (pl: T) {
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
            assert_eq!(Ok(()), param.set_cim(arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]])));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ])));
        }
    }

    let data = trajectory_generator(&net, 100, 200.0);
    let (CIM, M, T) = pl.fit(&net, &data, 0, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [1, 3, 3]);
    assert!(CIM.abs_diff_eq(&arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]]), 0.2));
}


#[test]
fn learn_ternary_cim_no_parents_MLE() {
    let mle = MLE{};
    learn_ternary_cim_no_parents(mle);
}


#[test]
fn learn_ternary_cim_no_parents_BA() {
    let ba = BayesianApproach{
        alpha: 1,
        tau: 1.0};
    learn_ternary_cim_no_parents(ba);
}


fn learn_mixed_discrete_cim<T: ParameterLearning> (pl: T) {
    let mut net = CtbnNetwork::init();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"),3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"),3))
        .unwrap();

    let n3 = net
        .add_node(generate_discrete_time_continous_node(String::from("n3"),4))
        .unwrap();
    net.add_edge(n1, n2);
    net.add_edge(n1, n3);
    net.add_edge(n2, n3);

    match &mut net.get_node_mut(n1).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[[[-3.0, 2.0, 1.0], 
                                  [1.5, -2.0, 0.5],
                                  [0.4, 0.6, -1.0]]])));
        }
    }

    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[
                [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
            ])));
        }
    }


    match &mut net.get_node_mut(n3).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[
                [[-1.0, 0.5, 0.3, 0.2], [0.5, -4.0, 2.5, 1.0], [2.5, 0.5, -4.0, 1.0], [0.7, 0.2, 0.1, -1.0]],
                [[-6.0, 2.0, 3.0, 1.0], [1.5, -3.0, 0.5, 1.0], [2.0, 1.3, -5.0 ,1.7], [2.5, 0.5, 1.0, -4.0]],
                [[-1.3, 0.3, 0.1, 0.9], [1.4, -4.0, 0.5, 2.1], [1.0, 1.5, -3.0, 0.5], [0.4, 0.3, 0.1, -0.8]],

                [[-2.0, 1.0, 0.7, 0.3], [1.3, -5.9, 2.7, 1.9], [2.0, 1.5, -4.0, 0.5], [0.2, 0.7, 0.1, -1.0]],
                [[-6.0, 1.0, 2.0, 3.0], [0.5, -3.0, 1.0, 1.5], [1.4, 2.1, -4.3, 0.8], [0.5, 1.0, 2.5, -4.0]],
                [[-1.3, 0.9, 0.3, 0.1], [0.1, -1.3, 0.2, 1.0], [0.5, 1.0, -3.0, 1.5], [0.1, 0.4, 0.3, -0.8]],

                [[-2.0, 1.0, 0.6, 0.4], [2.6, -7.1, 1.4, 3.1], [5.0, 1.0, -8.0, 2.0], [1.4, 0.4, 0.2, -2.0]],
                [[-3.0, 1.0, 1.5, 0.5], [3.0, -6.0, 1.0, 2.0], [0.3, 0.5, -1.9, 1.1], [5.0, 1.0, 2.0, -8.0]],
                [[-2.6, 0.6, 0.2, 1.8], [2.0, -6.0, 3.0, 1.0], [0.1, 0.5, -1.3, 0.7], [0.8, 0.6, 0.2, -1.6]],
            ])));
        }
    }


    let data = trajectory_generator(&net, 300, 300.0);
    let (CIM, M, T) = pl.fit(&net, &data, 2, None);
    print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
    assert_eq!(CIM.shape(), [9, 4, 4]);
    assert!(CIM.abs_diff_eq(&arr3(&[
                [[-1.0, 0.5, 0.3, 0.2], [0.5, -4.0, 2.5, 1.0], [2.5, 0.5, -4.0, 1.0], [0.7, 0.2, 0.1, -1.0]],
                [[-6.0, 2.0, 3.0, 1.0], [1.5, -3.0, 0.5, 1.0], [2.0, 1.3, -5.0 , 1.7], [2.5, 0.5, 1.0, -4.0]],
                [[-1.3, 0.3, 0.1, 0.9], [1.4, -4.0, 0.5, 2.1], [1.0, 1.5, -3.0, 0.5], [0.4, 0.3, 0.1, -0.8]],

                [[-2.0, 1.0, 0.7, 0.3], [1.3, -5.9, 2.7, 1.9], [2.0, 1.5, -4.0, 0.5], [0.2, 0.7, 0.1, -1.0]],
                [[-6.0, 1.0, 2.0, 3.0], [0.5, -3.0, 1.0, 1.5], [1.4, 2.1, -4.3, 0.8], [0.5, 1.0, 2.5, -4.0]],
                [[-1.3, 0.9, 0.3, 0.1], [0.1, -1.3, 0.2, 1.0], [0.5, 1.0, -3.0, 1.5], [0.1, 0.4, 0.3, -0.8]],

                [[-2.0, 1.0, 0.6, 0.4], [2.6, -7.1, 1.4, 3.1], [5.0, 1.0, -8.0, 2.0], [1.4, 0.4, 0.2, -2.0]],
                [[-3.0, 1.0, 1.5, 0.5], [3.0, -6.0, 1.0, 2.0], [0.3, 0.5, -1.9, 1.1], [5.0, 1.0, 2.0, -8.0]],
                [[-2.6, 0.6, 0.2, 1.8], [2.0, -6.0, 3.0, 1.0], [0.1, 0.5, -1.3, 0.7], [0.8, 0.6, 0.2, -1.6]],
            ]), 0.2));
}

#[test]
fn learn_mixed_discrete_cim_MLE() {
    let mle = MLE{};
    learn_mixed_discrete_cim(mle);
}


#[test]
fn learn_mixed_discrete_cim_BA() {
    let ba = BayesianApproach{
        alpha: 1,
        tau: 1.0};
    learn_mixed_discrete_cim(ba);
}
