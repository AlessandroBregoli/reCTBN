use ndarray::{arr1, arr2, arr3};
use reCTBN::process::ctbn::*;
use reCTBN::process::NetworkProcess;
use reCTBN::params;
use reCTBN::tools::*;

#[macro_use]
extern crate approx;

mod utils;

#[test]
fn run_sampling() {
    #![allow(unused_must_use)]
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(utils::generate_discrete_time_continous_node(
            String::from("n1"),
            2,
        ))
        .unwrap();
    let n2 = net
        .add_node(utils::generate_discrete_time_continous_node(
            String::from("n2"),
            2,
        ))
        .unwrap();
    net.add_edge(n1, n2);

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.set_cim(arr3(&[
                [
                    [-3.0, 3.0],
                    [2.0, -2.0]
                ],
            ]));
        }
    }

    match &mut net.get_node_mut(n2) {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.set_cim(arr3(&[
                [
                    [-1.0, 1.0],
                    [4.0, -4.0]
                ],
                [
                    [-6.0, 6.0],
                    [2.0, -2.0]
                ],
            ]));
        }
    }

    let data = trajectory_generator(&net, 4, 1.0, Some(6347747169756259));

    assert_eq!(4, data.get_trajectories().len());
    assert_relative_eq!(
        1.0,
        data.get_trajectories()[0].get_time()[data.get_trajectories()[0].get_time().len() - 1]
    );
}

#[test]
#[should_panic]
fn trajectory_wrong_shape() {
    let time = arr1(&[0.0, 0.2]);
    let events = arr2(&[[0, 3]]);
    Trajectory::new(time, events);
}

#[test]
#[should_panic]
fn dataset_wrong_shape() {
    let time = arr1(&[0.0, 0.2]);
    let events = arr2(&[[0, 3], [1, 2]]);
    let t1 = Trajectory::new(time, events);

    let time = arr1(&[0.0, 0.2]);
    let events = arr2(&[[0, 3, 3], [1, 2, 3]]);
    let t2 = Trajectory::new(time, events);
    Dataset::new(vec![t1, t2]);
}
