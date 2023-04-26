#![allow(non_snake_case)]

mod utils;
use ndarray::arr3;
use reCTBN::parameter_learning::*;
use reCTBN::params;
use reCTBN::params::Params::DiscreteStatesContinousTime;
use reCTBN::process::ctbn::*;
use reCTBN::process::NetworkProcess;
use reCTBN::tools::*;
use utils::*;

extern crate approx;
use crate::approx::AbsDiffEq;

fn learn_binary_cim<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(Ok(()), param.set_cim(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]])));
        }
    }

    match &mut net.get_node_mut(n2) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[
                    [[-1.0, 1.0], [4.0, -4.0]],
                    [[-6.0, 6.0], [2.0, -2.0]],
                ]))
            );
        }
    }

    let data = trajectory_generator(&net, 100, 100.0, Some(6347747169756259));
    let p = match pl.fit(&net, &data, 1, None) {
        params::Params::DiscreteStatesContinousTime(p) => p,
    };
    assert_eq!(p.get_cim().as_ref().unwrap().shape(), [2, 2, 2]);
    assert!(p.get_cim().as_ref().unwrap().abs_diff_eq(
        &arr3(&[[[-1.0, 1.0], [4.0, -4.0]], [[-6.0, 6.0], [2.0, -2.0]],]),
        0.1
    ));
}

fn generate_nodes(
    net: &mut CtbnNetwork,
    nodes_cardinality: usize,
    nodes_domain_cardinality: usize,
) {
    for node_label in 0..nodes_cardinality {
        net.add_node(generate_discrete_time_continous_node(
            node_label.to_string(),
            nodes_domain_cardinality,
        ))
        .unwrap();
    }
}

fn learn_binary_cim_gen<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 2);

    net.add_edge(0, 1);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..6.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let p_gen = match net.get_node(1) {
        DiscreteStatesContinousTime(p_gen) => p_gen,
    };

    let data = trajectory_generator(&net, 100, 100.0, Some(6347747169756259));
    let p_tj = match pl.fit(&net, &data, 1, None) {
        DiscreteStatesContinousTime(p_tj) => p_tj,
    };

    assert_eq!(
        p_tj.get_cim().as_ref().unwrap().shape(),
        p_gen.get_cim().as_ref().unwrap().shape()
    );
    assert!(p_tj
        .get_cim()
        .as_ref()
        .unwrap()
        .abs_diff_eq(&p_gen.get_cim().as_ref().unwrap(), 0.1));
}

#[test]
fn learn_binary_cim_MLE() {
    let mle = MLE {};
    learn_binary_cim(mle);
}

#[test]
fn learn_binary_cim_MLE_gen() {
    let mle = MLE {};
    learn_binary_cim_gen(mle);
}

#[test]
fn learn_binary_cim_BA() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_binary_cim(ba);
}

#[test]
fn learn_binary_cim_BA_gen() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_binary_cim_gen(ba);
}

fn learn_ternary_cim<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 3))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[[
                    [-3.0, 2.0, 1.0],
                    [1.5, -2.0, 0.5],
                    [0.4, 0.6, -1.0]
                ],]))
            );
        }
    }

    match &mut net.get_node_mut(n2) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[
                    [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                    [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                    [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
                ]))
            );
        }
    }

    let data = trajectory_generator(&net, 100, 200.0, Some(6347747169756259));
    let p = match pl.fit(&net, &data, 1, None) {
        params::Params::DiscreteStatesContinousTime(p) => p,
    };
    assert_eq!(p.get_cim().as_ref().unwrap().shape(), [3, 3, 3]);
    assert!(p.get_cim().as_ref().unwrap().abs_diff_eq(
        &arr3(&[
            [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
            [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
            [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
        ]),
        0.1
    ));
}

fn learn_ternary_cim_gen<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 3);

    net.add_edge(0, 1);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(4.0..6.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let p_gen = match net.get_node(1) {
        DiscreteStatesContinousTime(p_gen) => p_gen,
    };

    let data = trajectory_generator(&net, 100, 200.0, Some(6347747169756259));
    let p_tj = match pl.fit(&net, &data, 1, None) {
        DiscreteStatesContinousTime(p_tj) => p_tj,
    };

    assert_eq!(
        p_tj.get_cim().as_ref().unwrap().shape(),
        p_gen.get_cim().as_ref().unwrap().shape()
    );
    assert!(p_tj
        .get_cim()
        .as_ref()
        .unwrap()
        .abs_diff_eq(&p_gen.get_cim().as_ref().unwrap(), 0.1));
}

#[test]
fn learn_ternary_cim_MLE() {
    let mle = MLE {};
    learn_ternary_cim(mle);
}

#[test]
fn learn_ternary_cim_MLE_gen() {
    let mle = MLE {};
    learn_ternary_cim_gen(mle);
}

#[test]
fn learn_ternary_cim_BA() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_ternary_cim(ba);
}

#[test]
fn learn_ternary_cim_BA_gen() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_ternary_cim_gen(ba);
}

fn learn_ternary_cim_no_parents<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 3))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[[
                    [-3.0, 2.0, 1.0],
                    [1.5, -2.0, 0.5],
                    [0.4, 0.6, -1.0]
                ]]))
            );
        }
    }

    match &mut net.get_node_mut(n2) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[
                    [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                    [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                    [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
                ]))
            );
        }
    }

    let data = trajectory_generator(&net, 100, 200.0, Some(6347747169756259));
    let p = match pl.fit(&net, &data, 0, None) {
        params::Params::DiscreteStatesContinousTime(p) => p,
    };
    assert_eq!(p.get_cim().as_ref().unwrap().shape(), [1, 3, 3]);
    assert!(p.get_cim().as_ref().unwrap().abs_diff_eq(
        &arr3(&[[[-3.0, 2.0, 1.0], [1.5, -2.0, 0.5], [0.4, 0.6, -1.0]],]),
        0.1
    ));
}

fn learn_ternary_cim_no_parents_gen<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 3);

    net.add_edge(0, 1);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..6.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let p_gen = match net.get_node(0) {
        DiscreteStatesContinousTime(p_gen) => p_gen,
    };

    let data = trajectory_generator(&net, 100, 200.0, Some(6347747169756259));
    let p_tj = match pl.fit(&net, &data, 0, None) {
        DiscreteStatesContinousTime(p_tj) => p_tj,
    };

    assert_eq!(
        p_tj.get_cim().as_ref().unwrap().shape(),
        p_gen.get_cim().as_ref().unwrap().shape()
    );
    assert!(p_tj
        .get_cim()
        .as_ref()
        .unwrap()
        .abs_diff_eq(&p_gen.get_cim().as_ref().unwrap(), 0.1));
}

#[test]
fn learn_ternary_cim_no_parents_MLE() {
    let mle = MLE {};
    learn_ternary_cim_no_parents(mle);
}

#[test]
fn learn_ternary_cim_no_parents_MLE_gen() {
    let mle = MLE {};
    learn_ternary_cim_no_parents_gen(mle);
}

#[test]
fn learn_ternary_cim_no_parents_BA() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_ternary_cim_no_parents(ba);
}

#[test]
fn learn_ternary_cim_no_parents_BA_gen() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_ternary_cim_no_parents_gen(ba);
}

fn learn_mixed_discrete_cim<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 3))
        .unwrap();

    let n3 = net
        .add_node(generate_discrete_time_continous_node(String::from("n3"), 4))
        .unwrap();
    net.add_edge(n1, n2);
    net.add_edge(n1, n3);
    net.add_edge(n2, n3);

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[[
                    [-3.0, 2.0, 1.0],
                    [1.5, -2.0, 0.5],
                    [0.4, 0.6, -1.0]
                ],]))
            );
        }
    }

    match &mut net.get_node_mut(n2) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[
                    [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                    [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                    [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
                ]))
            );
        }
    }

    match &mut net.get_node_mut(n3) {
        params::Params::DiscreteStatesContinousTime(param) => {
            assert_eq!(
                Ok(()),
                param.set_cim(arr3(&[
                    [
                        [-1.0, 0.5, 0.3, 0.2],
                        [0.5, -4.0, 2.5, 1.0],
                        [2.5, 0.5, -4.0, 1.0],
                        [0.7, 0.2, 0.1, -1.0]
                    ],
                    [
                        [-6.0, 2.0, 3.0, 1.0],
                        [1.5, -3.0, 0.5, 1.0],
                        [2.0, 1.3, -5.0, 1.7],
                        [2.5, 0.5, 1.0, -4.0]
                    ],
                    [
                        [-1.3, 0.3, 0.1, 0.9],
                        [1.4, -4.0, 0.5, 2.1],
                        [1.0, 1.5, -3.0, 0.5],
                        [0.4, 0.3, 0.1, -0.8]
                    ],
                    [
                        [-2.0, 1.0, 0.7, 0.3],
                        [1.3, -5.9, 2.7, 1.9],
                        [2.0, 1.5, -4.0, 0.5],
                        [0.2, 0.7, 0.1, -1.0]
                    ],
                    [
                        [-6.0, 1.0, 2.0, 3.0],
                        [0.5, -3.0, 1.0, 1.5],
                        [1.4, 2.1, -4.3, 0.8],
                        [0.5, 1.0, 2.5, -4.0]
                    ],
                    [
                        [-1.3, 0.9, 0.3, 0.1],
                        [0.1, -1.3, 0.2, 1.0],
                        [0.5, 1.0, -3.0, 1.5],
                        [0.1, 0.4, 0.3, -0.8]
                    ],
                    [
                        [-2.0, 1.0, 0.6, 0.4],
                        [2.6, -7.1, 1.4, 3.1],
                        [5.0, 1.0, -8.0, 2.0],
                        [1.4, 0.4, 0.2, -2.0]
                    ],
                    [
                        [-3.0, 1.0, 1.5, 0.5],
                        [3.0, -6.0, 1.0, 2.0],
                        [0.3, 0.5, -1.9, 1.1],
                        [5.0, 1.0, 2.0, -8.0]
                    ],
                    [
                        [-2.6, 0.6, 0.2, 1.8],
                        [2.0, -6.0, 3.0, 1.0],
                        [0.1, 0.5, -1.3, 0.7],
                        [0.8, 0.6, 0.2, -1.6]
                    ],
                ]))
            );
        }
    }

    let data = trajectory_generator(&net, 300, 300.0, Some(6347747169756259));
    let p = match pl.fit(&net, &data, 2, None) {
        params::Params::DiscreteStatesContinousTime(p) => p,
    };
    assert_eq!(p.get_cim().as_ref().unwrap().shape(), [9, 4, 4]);
    assert!(p.get_cim().as_ref().unwrap().abs_diff_eq(
        &arr3(&[
            [
                [-1.0, 0.5, 0.3, 0.2],
                [0.5, -4.0, 2.5, 1.0],
                [2.5, 0.5, -4.0, 1.0],
                [0.7, 0.2, 0.1, -1.0]
            ],
            [
                [-6.0, 2.0, 3.0, 1.0],
                [1.5, -3.0, 0.5, 1.0],
                [2.0, 1.3, -5.0, 1.7],
                [2.5, 0.5, 1.0, -4.0]
            ],
            [
                [-1.3, 0.3, 0.1, 0.9],
                [1.4, -4.0, 0.5, 2.1],
                [1.0, 1.5, -3.0, 0.5],
                [0.4, 0.3, 0.1, -0.8]
            ],
            [
                [-2.0, 1.0, 0.7, 0.3],
                [1.3, -5.9, 2.7, 1.9],
                [2.0, 1.5, -4.0, 0.5],
                [0.2, 0.7, 0.1, -1.0]
            ],
            [
                [-6.0, 1.0, 2.0, 3.0],
                [0.5, -3.0, 1.0, 1.5],
                [1.4, 2.1, -4.3, 0.8],
                [0.5, 1.0, 2.5, -4.0]
            ],
            [
                [-1.3, 0.9, 0.3, 0.1],
                [0.1, -1.3, 0.2, 1.0],
                [0.5, 1.0, -3.0, 1.5],
                [0.1, 0.4, 0.3, -0.8]
            ],
            [
                [-2.0, 1.0, 0.6, 0.4],
                [2.6, -7.1, 1.4, 3.1],
                [5.0, 1.0, -8.0, 2.0],
                [1.4, 0.4, 0.2, -2.0]
            ],
            [
                [-3.0, 1.0, 1.5, 0.5],
                [3.0, -6.0, 1.0, 2.0],
                [0.3, 0.5, -1.9, 1.1],
                [5.0, 1.0, 2.0, -8.0]
            ],
            [
                [-2.6, 0.6, 0.2, 1.8],
                [2.0, -6.0, 3.0, 1.0],
                [0.1, 0.5, -1.3, 0.7],
                [0.8, 0.6, 0.2, -1.6]
            ],
        ]),
        0.2
    ));
}

fn learn_mixed_discrete_cim_gen<T: ParameterLearning>(pl: T) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 3);
    net.add_node(generate_discrete_time_continous_node(String::from("3"), 4))
        .unwrap();
    net.add_edge(0, 1);
    net.add_edge(0, 2);
    net.add_edge(1, 2);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..8.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let p_gen = match net.get_node(2) {
        DiscreteStatesContinousTime(p_gen) => p_gen,
    };

    let data = trajectory_generator(&net, 300, 300.0, Some(6347747169756259));
    let p_tj = match pl.fit(&net, &data, 2, None) {
        DiscreteStatesContinousTime(p_tj) => p_tj,
    };

    assert_eq!(
        p_tj.get_cim().as_ref().unwrap().shape(),
        p_gen.get_cim().as_ref().unwrap().shape()
    );
    assert!(p_tj
        .get_cim()
        .as_ref()
        .unwrap()
        .abs_diff_eq(&p_gen.get_cim().as_ref().unwrap(), 0.2));
}

#[test]
fn learn_mixed_discrete_cim_MLE() {
    let mle = MLE {};
    learn_mixed_discrete_cim(mle);
}

#[test]
fn learn_mixed_discrete_cim_MLE_gen() {
    let mle = MLE {};
    learn_mixed_discrete_cim_gen(mle);
}

#[test]
fn learn_mixed_discrete_cim_BA() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_mixed_discrete_cim(ba);
}

#[test]
fn learn_mixed_discrete_cim_BA_gen() {
    let ba = BayesianApproach {
        alpha: 1,
        tau: Tau::Constant(1.0),
    };
    learn_mixed_discrete_cim_gen(ba);
}
