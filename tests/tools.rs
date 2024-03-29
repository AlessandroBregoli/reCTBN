use std::ops::Range;

use ndarray::{arr1, arr2, arr3};
use reCTBN::params::ParamsTrait;
use reCTBN::process::ctbn::*;
use reCTBN::process::ctmp::*;
use reCTBN::process::NetworkProcess;
use reCTBN::params;
use reCTBN::tools::*;

use utils::*;

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

#[test]
#[should_panic]
fn uniform_graph_generator_wrong_density_1() {
    let density = 2.1;
    let _structure_generator: UniformGraphGenerator = RandomGraphGenerator::new(
        density,
        None
    );
}

#[test]
#[should_panic]
fn uniform_graph_generator_wrong_density_2() {
    let density = -0.5;
    let _structure_generator: UniformGraphGenerator = RandomGraphGenerator::new(
        density,
        None
    );
}

#[test]
fn uniform_graph_generator_right_densities() {
    for density in [1.0, 0.75, 0.5, 0.25, 0.0] {
        let _structure_generator: UniformGraphGenerator = RandomGraphGenerator::new(
            density,
            None
        );
    }
}

#[test]
fn uniform_graph_generator_generate_graph_ctbn() {
    let mut net = CtbnNetwork::new();
    let nodes_cardinality = 0..=100;
    let nodes_domain_cardinality = 2;
    for node_label in nodes_cardinality {
        net.add_node(
            utils::generate_discrete_time_continous_node(
                node_label.to_string(),
                nodes_domain_cardinality,
            )
        ).unwrap();
    }
    let density = 1.0/3.0;
    let mut structure_generator: UniformGraphGenerator = RandomGraphGenerator::new(
        density,
        Some(7641630759785120)
    );
    structure_generator.generate_graph(&mut net);
    let mut edges = 0;
    for node in net.get_node_indices(){
        edges += net.get_children_set(node).len()
    }
    let nodes = net.get_node_indices().len() as f64;
    let expected_edges = (density * nodes * (nodes - 1.0)).round() as usize;
    let tolerance = ((expected_edges as f64)*0.05) as usize; // ±5% of tolerance
    // As the way `generate_graph()` is implemented we can only reasonably
    // expect the number of edges to be somewhere around the expected value.
    assert!((expected_edges - tolerance) <= edges && edges <= (expected_edges + tolerance));
}

#[test]
#[should_panic]
fn uniform_graph_generator_generate_graph_ctmp() {
    let mut net = CtmpProcess::new();
    let node_label = String::from("0");
    let node_domain_cardinality = 4;
    net.add_node(
        generate_discrete_time_continous_node(
            node_label,
            node_domain_cardinality
        )
    ).unwrap();
    let density = 1.0/3.0;
    let mut structure_generator: UniformGraphGenerator = RandomGraphGenerator::new(
        density,
        Some(7641630759785120)
    );
    structure_generator.generate_graph(&mut net);
}

#[test]
#[should_panic]
fn uniform_parameters_generator_wrong_density_1() {
    let interval: Range<f64> = -2.0..-5.0;
    let _cim_generator: UniformParametersGenerator = RandomParametersGenerator::new(
        interval,
        None
    );
}

#[test]
#[should_panic]
fn uniform_parameters_generator_wrong_density_2() {
    let interval: Range<f64> = -1.0..0.0;
    let _cim_generator: UniformParametersGenerator = RandomParametersGenerator::new(
        interval,
        None
    );
}

#[test]
fn uniform_parameters_generator_right_densities_ctbn() {
    let mut net = CtbnNetwork::new();
    let nodes_cardinality = 0..=3;
    let nodes_domain_cardinality = 9;
    for node_label in nodes_cardinality {
        net.add_node(
            generate_discrete_time_continous_node(
                node_label.to_string(),
                nodes_domain_cardinality,
            )
        ).unwrap();
    }
    let density = 1.0/3.0;
    let seed = Some(7641630759785120);
    let interval = 0.0..7.0;
    let mut structure_generator: UniformGraphGenerator = RandomGraphGenerator::new(
        density,
        seed
    );
    structure_generator.generate_graph(&mut net);
    let mut cim_generator: UniformParametersGenerator = RandomParametersGenerator::new(
        interval,
        seed
    );
    cim_generator.generate_parameters(&mut net);
    for node in net.get_node_indices() {
        assert_eq!(
            Ok(()),
            net.get_node(node).validate_params()
        );
    }
}

#[test]
fn uniform_parameters_generator_right_densities_ctmp() {
    let mut net = CtmpProcess::new();
    let node_label = String::from("0");
    let node_domain_cardinality = 4;
    net.add_node(
        generate_discrete_time_continous_node(
            node_label,
            node_domain_cardinality
        )
    ).unwrap();
    let seed = Some(7641630759785120);
    let interval = 0.0..7.0;
    let mut cim_generator: UniformParametersGenerator = RandomParametersGenerator::new(
        interval,
        seed
    );
    cim_generator.generate_parameters(&mut net);
    for node in net.get_node_indices() {
        assert_eq!(
            Ok(()),
            net.get_node(node).validate_params()
        );
    }
}
