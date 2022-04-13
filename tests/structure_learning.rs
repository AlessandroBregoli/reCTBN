
mod utils;
use utils::*;

use rustyCTBN::ctbn::*;
use rustyCTBN::network::Network;
use rustyCTBN::tools::*;
use rustyCTBN::structure_learning::*;
use ndarray::{arr1, arr2};
use std::collections::BTreeSet;


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
