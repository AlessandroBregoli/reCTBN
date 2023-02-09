#![allow(non_snake_case)]

use std::collections::BTreeSet;
use std::time::Duration;
use criterion::{criterion_group, criterion_main, Criterion};

use reCTBN::params::DiscreteStatesContinousTimeParams;
use reCTBN::params::Params::DiscreteStatesContinousTime;
use reCTBN::process::NetworkProcess;
use reCTBN::parameter_learning::BayesianApproach;
use reCTBN::process::ctbn::CtbnNetwork;
use reCTBN::structure_learning::constraint_based_algorithm::CTPC;
use reCTBN::structure_learning::hypothesis_test::{ChiSquare, F};
use reCTBN::structure_learning::StructureLearningAlgorithm;
use reCTBN::tools::trajectory_generator;
use reCTBN::tools::Dataset;
use reCTBN::tools::RandomGraphGenerator;
use reCTBN::tools::RandomParametersGenerator;
use reCTBN::tools::UniformGraphGenerator;
use reCTBN::tools::UniformParametersGenerator;


fn uniform_parameters_generator_right_densities_ctmp() -> (CtbnNetwork, Dataset) {
    let mut net = CtbnNetwork::new();
    let nodes_cardinality = 5;
    let domain_cardinality = 3;
    for node in 0..nodes_cardinality {
        // Create the domain for a discrete node
        let mut domain = BTreeSet::new();
        for dvalue in 0..domain_cardinality {
            domain.insert(dvalue.to_string());
        }
        // Create the parameters for a discrete node using the domain
        let param = DiscreteStatesContinousTimeParams::new(node.to_string(), domain);
        //Create the node using the parameters
        let node = DiscreteStatesContinousTime(param);
        // Add the node to the network
        net.add_node(node).unwrap();
    }

    // Initialize the Graph Generator using the one with an
    // uniform distribution
    let mut structure_generator = UniformGraphGenerator::new(1.0 / 3.0, Some(7641630759785120));

    // Generate the graph directly on the network
    structure_generator.generate_graph(&mut net);

    // Initialize the parameters generator with uniform distributin
    let mut cim_generator = UniformParametersGenerator::new(3.0..7.0, Some(7641630759785120));

    // Generate CIMs with uniformly distributed parameters.
    cim_generator.generate_parameters(&mut net);

    let dataset = trajectory_generator(&net, 300, 200.0, Some(30230423));

    return (net, dataset);
}

fn structure_learning_CTPC(net: CtbnNetwork, dataset: &Dataset) {
    // Initialize the hypothesis tests to pass to the CTPC with their
    // respective significance level `alpha`
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    // Use the bayesian approach to learn the parameters
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    //Initialize CTPC
    let ctpc = CTPC::new(parameter_learning, f, chi_sq);
    // Learn the structure of the network from the generated trajectory
    ctpc.fit_transform(net, dataset);
}

pub fn criterion_benchmark_ctpc(c: &mut Criterion) {
    let mut group = c.benchmark_group("structure_learning_CTPC");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10).measurement_time(Duration::from_secs(20));
    group.bench_function("CTPC", move |b| {
        b.iter_batched(
            || uniform_parameters_generator_right_densities_ctmp(),
            |(net, dataset)| structure_learning_CTPC(net, &dataset),
            criterion::BatchSize::PerIteration,
        )
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark_ctpc);
criterion_main!(benches);
