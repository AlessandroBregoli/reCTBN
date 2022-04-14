
mod utils;
use utils::*;

use rustyCTBN::ctbn::*;
use rustyCTBN::network::Network;
use rustyCTBN::tools::*;
use rustyCTBN::structure_learning::score_function::*;
use rustyCTBN::structure_learning::score_based_algorithm::*;
use rustyCTBN::structure_learning::StructureLearningAlgorithm;
use ndarray::{arr1, arr2, arr3};
use std::collections::BTreeSet;
use rustyCTBN::params;


#[macro_use]
extern crate approx;

#[test]
fn simple_log_likelihood() {
    let mut net = CtbnNetwork::init();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"),2))
        .unwrap();

    let trj = Trajectory{
        time: arr1(&[0.0,0.1,0.3]),
        events: arr2(&[[0],[1],[1]])};

    let dataset = Dataset{
        trajectories: vec![trj]};

    let ll = LogLikelihood::init(1, 1.0);

    assert_abs_diff_eq!(0.04257, ll.call(&net, n1, &BTreeSet::new(), &dataset), epsilon=1e-3);

}


#[test]
fn simple_bic() {
    let mut net = CtbnNetwork::init();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"),2))
        .unwrap();

    let trj = Trajectory{
        time: arr1(&[0.0,0.1,0.3]),
        events: arr2(&[[0],[1],[1]])};

    let dataset = Dataset{
        trajectories: vec![trj]};

    let ll = BIC::init(1, 1.0);

    assert_abs_diff_eq!(-0.65058, ll.call(&net, n1, &BTreeSet::new(), &dataset), epsilon=1e-3);

}

fn learn_ternary_net_2_nodes<T: StructureLearningAlgorithm> (sl: T) {
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

    let data = trajectory_generator(&net, 100, 20.0, Some(6347747169756259),);

    let net = sl.fit_transform(net, &data);
    assert_eq!(BTreeSet::from_iter(vec![n1]), net.get_parent_set(n2));
    assert_eq!(BTreeSet::new(), net.get_parent_set(n1));
}


#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing_ll() {
    let ll = LogLikelihood::init(1, 1.0);
    let hl = HillClimbing::init(ll, None);
    learn_ternary_net_2_nodes(hl);
}

#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing_bic() {
    let bic = BIC::init(1, 1.0);
    let hl = HillClimbing::init(bic, None);
    learn_ternary_net_2_nodes(hl);
}



fn learn_mixed_discrete_net_3_nodes<T: StructureLearningAlgorithm> (sl: T) {
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


    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259),);
    let net = sl.fit_transform(net, &data);

    assert_eq!(BTreeSet::new(), net.get_parent_set(n1));
    assert_eq!(BTreeSet::from_iter(vec![n1]), net.get_parent_set(n2));
    assert_eq!(BTreeSet::from_iter(vec![n1, n2]), net.get_parent_set(n3));
}


#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_ll() {
    let ll = LogLikelihood::init(1, 1.0);
    let hl = HillClimbing::init(ll, None);
    learn_mixed_discrete_net_3_nodes(hl);
}

#[test]
pub fn learn_mixed_discrete_net_3_nodes_hill_climbing_bic() {
    let bic = BIC::init(1, 1.0);
    let hl = HillClimbing::init(bic, None);
    learn_mixed_discrete_net_3_nodes(hl);
}
