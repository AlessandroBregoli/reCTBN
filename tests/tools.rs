
use rustyCTBN::tools::*;
use rustyCTBN::network::Network;
use rustyCTBN::ctbn::*;
use rustyCTBN::node;
use rustyCTBN::params;
use std::collections::BTreeSet;
use ndarray::arr3;



#[macro_use]
extern crate approx;

mod utils;

#[test]
fn run_sampling() {
    let mut net = CtbnNetwork::init();
    let n1 = net.add_node(utils::generate_discrete_time_continous_node(String::from("n1"),2)).unwrap();
    let n2 = net.add_node(utils::generate_discrete_time_continous_node(String::from("n2"),2)).unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.set_cim(arr3(&[[[-3.0,3.0],[2.0,-2.0]]]));
        }
    }


    match &mut net.get_node_mut(n2).params {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.set_cim(arr3(&[
                                     [[-1.0,1.0],[4.0,-4.0]],
                                     [[-6.0,6.0],[2.0,-2.0]]]));
        }
    }

    let data = trajectory_generator(&net, 4, 1.0, Some(1234),);

    assert_eq!(4, data.trajectories.len());
    assert_relative_eq!(1.0, data.trajectories[0].time[data.trajectories[0].time.len()-1]);
}


