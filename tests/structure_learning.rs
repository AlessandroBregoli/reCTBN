
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

    let data = trajectory_generator(&net, 100, 200.0, Some(6347747169756259),);

    let net = sl.call(net, &data);
    assert_eq!(BTreeSet::from_iter(vec![n1]), net.get_parent_set(n2));
    assert_eq!(BTreeSet::new(), net.get_parent_set(n1));
}

#[test]
pub fn learn_ternary_net_2_nodes_hill_climbing() {
    let bic = BIC::init(1, 1.0);
    let hl = HillClimbing::init(bic);
    learn_ternary_net_2_nodes(hl);
}
