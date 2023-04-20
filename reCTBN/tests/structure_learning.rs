#![allow(non_snake_case)]

mod utils;
use std::collections::BTreeSet;

use ndarray::{arr1, arr2, arr3};
use reCTBN::parameter_learning::BayesianApproach;
use reCTBN::params;
use reCTBN::process::ctbn::*;
use reCTBN::process::NetworkProcess;
use reCTBN::structure_learning::constraint_based_algorithm::*;
use reCTBN::structure_learning::hypothesis_test::*;
use reCTBN::structure_learning::score_based_algorithm::*;
use reCTBN::structure_learning::score_function::*;
use reCTBN::structure_learning::StructuralLearningAlgorithm;
use reCTBN::tools::*;
use utils::*;

#[macro_use]
extern crate approx;

#[test]
fn simple_score_test() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    let trj = Trajectory::new(arr1(&[0.0, 0.1, 0.3]), arr2(&[[0], [1], [1]]));

    let dataset = Dataset::new(vec![trj]);

    let ll = LogLikelihood::new(1, 1.0);

    assert_abs_diff_eq!(
        0.04257,
        ll.call(&net, n1, &BTreeSet::new(), &dataset),
        epsilon = 1e-3
    );
}

#[test]
fn simple_bic() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    let trj = Trajectory::new(arr1(&[0.0, 0.1, 0.3]), arr2(&[[0], [1], [1]]));

    let dataset = Dataset::new(vec![trj]);
    let bic = BIC::new(1, 1.0);

    assert_abs_diff_eq!(
        -0.65058,
        bic.call(&net, n1, &BTreeSet::new(), &dataset),
        epsilon = 1e-3
    );
}

fn check_compatibility_between_dataset_and_network<T: StructuralLearningAlgorithm>(sl: T) {
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

    let data = trajectory_generator(&net, 100, 30.0, Some(6347747169756259));

    let mut net = CtbnNetwork::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 3))
        .unwrap();
    let _net = sl.fit_transform(net, &data);
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

fn check_compatibility_between_dataset_and_network_gen<T: StructuralLearningAlgorithm>(sl: T) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 3);
    net.add_node(generate_discrete_time_continous_node(String::from("3"), 4))
        .unwrap();

    net.add_edge(0, 1);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(0.0..7.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 100, 30.0, Some(6347747169756259));

    let mut net = CtbnNetwork::new();
    let _n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("0"), 3))
        .unwrap();
    let _net = sl.fit_transform(net, &data);
}

#[test]
#[should_panic]
pub fn check_compatibility_between_dataset_and_network_hill_climbing() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, None);
    check_compatibility_between_dataset_and_network(hl);
}

#[test]
#[should_panic]
pub fn check_compatibility_between_dataset_and_network_hill_climbing_gen() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, None);
    check_compatibility_between_dataset_and_network_gen(hl);
}

fn learn_ternary_net_2_nodes<T: StructuralLearningAlgorithm>(sl: T) {
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

    let data = trajectory_generator(&net, 100, 20.0, Some(6347747169756259));

    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::from_iter(vec![n1]), net.get_parent_set(n2));
    assert_eq!(BTreeSet::new(), net.get_parent_set(n1));
}

fn learn_ternary_net_2_nodes_gen<T: StructuralLearningAlgorithm>(sl: T) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 3);

    net.add_edge(0, 1);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(0.0..7.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 100, 20.0, Some(6347747169756259));

    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
    assert_eq!(BTreeSet::new(), net.get_parent_set(0));
}

#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing_ll() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, None);
    learn_ternary_net_2_nodes(hl);
}

#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing_ll_gen() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, None);
    learn_ternary_net_2_nodes_gen(hl);
}

#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing_bic() {
    let bic = BIC::new(1, 1.0);
    let hl = HillClimbing::new(bic, None);
    learn_ternary_net_2_nodes(hl);
}

#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing_bic_gen() {
    let bic = BIC::new(1, 1.0);
    let hl = HillClimbing::new(bic, None);
    learn_ternary_net_2_nodes_gen(hl);
}

fn get_mixed_discrete_net_3_nodes_with_data() -> (CtbnNetwork, Dataset) {
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

    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259));
    return (net, data);
}

fn get_mixed_discrete_net_3_nodes_with_data_gen() -> (CtbnNetwork, Dataset) {
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 2, 3);
    net.add_node(generate_discrete_time_continous_node(String::from("3"), 4))
        .unwrap();

    net.add_edge(0, 1);
    net.add_edge(0, 2);
    net.add_edge(1, 2);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(0.0..7.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259));
    return (net, data);
}

fn learn_mixed_discrete_net_3_nodes<T: StructuralLearningAlgorithm>(sl: T) {
    let (net, data) = get_mixed_discrete_net_3_nodes_with_data();
    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::new(), net.get_parent_set(0));
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
    assert_eq!(BTreeSet::from_iter(vec![0, 1]), net.get_parent_set(2));
}

fn learn_mixed_discrete_net_3_nodes_gen<T: StructuralLearningAlgorithm>(sl: T) {
    let (net, data) = get_mixed_discrete_net_3_nodes_with_data_gen();
    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::new(), net.get_parent_set(0));
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
    assert_eq!(BTreeSet::from_iter(vec![0, 1]), net.get_parent_set(2));
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_ll() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, None);
    learn_mixed_discrete_net_3_nodes(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_ll_gen() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, None);
    learn_mixed_discrete_net_3_nodes_gen(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_bic() {
    let bic = BIC::new(1, 1.0);
    let hl = HillClimbing::new(bic, None);
    learn_mixed_discrete_net_3_nodes(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_bic_gen() {
    let bic = BIC::new(1, 1.0);
    let hl = HillClimbing::new(bic, None);
    learn_mixed_discrete_net_3_nodes_gen(hl);
}

fn learn_mixed_discrete_net_3_nodes_1_parent_constraint<T: StructuralLearningAlgorithm>(sl: T) {
    let (net, data) = get_mixed_discrete_net_3_nodes_with_data();
    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::new(), net.get_parent_set(0));
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(2));
}

fn learn_mixed_discrete_net_3_nodes_1_parent_constraint_gen<T: StructuralLearningAlgorithm>(sl: T) {
    let (net, data) = get_mixed_discrete_net_3_nodes_with_data_gen();
    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::new(), net.get_parent_set(0));
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(1));
    assert_eq!(BTreeSet::from_iter(vec![0]), net.get_parent_set(2));
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_ll_1_parent_constraint() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, Some(1));
    learn_mixed_discrete_net_3_nodes_1_parent_constraint(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_ll_1_parent_constraint_gen() {
    let ll = LogLikelihood::new(1, 1.0);
    let hl = HillClimbing::new(ll, Some(1));
    learn_mixed_discrete_net_3_nodes_1_parent_constraint_gen(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_bic_1_parent_constraint() {
    let bic = BIC::new(1, 1.0);
    let hl = HillClimbing::new(bic, Some(1));
    learn_mixed_discrete_net_3_nodes_1_parent_constraint(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_bic_1_parent_constraint_gen() {
    let bic = BIC::new(1, 1.0);
    let hl = HillClimbing::new(bic, Some(1));
    learn_mixed_discrete_net_3_nodes_1_parent_constraint_gen(hl);
}

#[test]
pub fn chi_square_compare_matrices() {
    let i: usize = 1;
    let M1 = arr3(&[
        [[0, 2, 3], [4, 0, 6], [7, 8, 0]],
        [[0, 12, 90], [3, 0, 40], [6, 40, 0]],
        [[0, 2, 3], [4, 0, 6], [44, 66, 0]],
    ]);
    let j: usize = 0;
    let M2 = arr3(&[[[0, 200, 300], [400, 0, 600], [700, 800, 0]]]);
    let chi_sq = ChiSquare::new(1e-4);
    assert!(!chi_sq.compare_matrices(i, &M1, j, &M2));
}

#[test]
pub fn chi_square_compare_matrices_2() {
    let i: usize = 1;
    let M1 = arr3(&[
        [[0, 2, 3], [4, 0, 6], [7, 8, 0]],
        [[0, 20, 30], [40, 0, 60], [70, 80, 0]],
        [[0, 2, 3], [4, 0, 6], [44, 66, 0]],
    ]);
    let j: usize = 0;
    let M2 = arr3(&[[[0, 200, 300], [400, 0, 600], [700, 800, 0]]]);
    let chi_sq = ChiSquare::new(1e-4);
    assert!(chi_sq.compare_matrices(i, &M1, j, &M2));
}

#[test]
pub fn chi_square_compare_matrices_3() {
    let i: usize = 1;
    let M1 = arr3(&[
        [[0, 2, 3], [4, 0, 6], [7, 8, 0]],
        [[0, 21, 31], [41, 0, 59], [71, 79, 0]],
        [[0, 2, 3], [4, 0, 6], [44, 66, 0]],
    ]);
    let j: usize = 0;
    let M2 = arr3(&[[[0, 200, 300], [400, 0, 600], [700, 800, 0]]]);
    let chi_sq = ChiSquare::new(1e-4);
    assert!(chi_sq.compare_matrices(i, &M1, j, &M2));
}

#[test]
pub fn chi_square_call() {
    let (net, data) = get_mixed_discrete_net_3_nodes_with_data();
    let N3: usize = 2;
    let N2: usize = 1;
    let N1: usize = 0;
    let mut separation_set = BTreeSet::new();
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let mut cache = Cache::new(&parameter_learning);
    let chi_sq = ChiSquare::new(1e-4);

    assert!(chi_sq.call(&net, N1, N3, &separation_set, &data, &mut cache));
    let mut cache = Cache::new(&parameter_learning);
    assert!(!chi_sq.call(&net, N3, N1, &separation_set, &data, &mut cache));
    assert!(!chi_sq.call(&net, N3, N2, &separation_set, &data, &mut cache));
    separation_set.insert(N1);
    let mut cache = Cache::new(&parameter_learning);
    assert!(chi_sq.call(&net, N2, N3, &separation_set, &data, &mut cache));
}

#[test]
pub fn f_call() {
    let (net, data) = get_mixed_discrete_net_3_nodes_with_data();
    let N3: usize = 2;
    let N2: usize = 1;
    let N1: usize = 0;
    let mut separation_set = BTreeSet::new();
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let mut cache = Cache::new(&parameter_learning);
    let f = F::new(1e-6);

    assert!(f.call(&net, N1, N3, &separation_set, &data, &mut cache));
    let mut cache = Cache::new(&parameter_learning);
    assert!(!f.call(&net, N3, N1, &separation_set, &data, &mut cache));
    assert!(!f.call(&net, N3, N2, &separation_set, &data, &mut cache));
    separation_set.insert(N1);
    let mut cache = Cache::new(&parameter_learning);
    assert!(f.call(&net, N2, N3, &separation_set, &data, &mut cache));
}

#[test]
pub fn learn_ternary_net_2_nodes_ctpc() {
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = CTPC::new(parameter_learning, f, chi_sq);
    learn_ternary_net_2_nodes(ctpc);
}

#[test]
pub fn learn_ternary_net_2_nodes_ctpc_gen() {
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = CTPC::new(parameter_learning, f, chi_sq);
    learn_ternary_net_2_nodes_gen(ctpc);
}

#[test]
fn learn_mixed_discrete_net_3_nodes_ctpc() {
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = CTPC::new(parameter_learning, f, chi_sq);
    learn_mixed_discrete_net_3_nodes(ctpc);
}

#[test]
fn learn_mixed_discrete_net_3_nodes_ctpc_gen() {
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = CTPC::new(parameter_learning, f, chi_sq);
    learn_mixed_discrete_net_3_nodes_gen(ctpc);
}
