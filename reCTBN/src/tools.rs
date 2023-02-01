//! Contains commonly used methods used across the crate.

use std::ops::{DivAssign, MulAssign, Range};

use ndarray::{Array, Array1, Array2, Array3, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::params::ParamsTrait;
use crate::process::NetworkProcess;
use crate::sampling::{ForwardSampler, Sampler};
use crate::{params, process};

#[derive(Clone)]
pub struct Trajectory {
    time: Array1<f64>,
    events: Array2<usize>,
}

impl Trajectory {
    pub fn new(time: Array1<f64>, events: Array2<usize>) -> Trajectory {
        //Events and time are two part of the same trajectory. For this reason they must have the
        //same number of sample.
        if time.shape()[0] != events.shape()[0] {
            panic!("time.shape[0] must be equal to events.shape[0]");
        }
        Trajectory { time, events }
    }

    pub fn get_time(&self) -> &Array1<f64> {
        &self.time
    }

    pub fn get_events(&self) -> &Array2<usize> {
        &self.events
    }
}

#[derive(Clone)]
pub struct Dataset {
    trajectories: Vec<Trajectory>,
}

impl Dataset {
    pub fn new(trajectories: Vec<Trajectory>) -> Dataset {
        //All the trajectories in the same dataset must represent the same process. For this reason
        //each trajectory must represent the same number of variables.
        if trajectories
            .iter()
            .any(|x| trajectories[0].get_events().shape()[1] != x.get_events().shape()[1])
        {
            panic!("All the trajectories mus represents the same number of variables");
        }
        Dataset { trajectories }
    }

    pub fn get_trajectories(&self) -> &Vec<Trajectory> {
        &self.trajectories
    }
}

pub fn trajectory_generator<T: process::NetworkProcess>(
    net: &T,
    n_trajectories: u64,
    t_end: f64,
    seed: Option<u64>,
) -> Dataset {
    //Tmp growing vector containing generated trajectories.
    let mut trajectories: Vec<Trajectory> = Vec::new();

    //Random Generator object

    let mut sampler = ForwardSampler::new(net, seed);
    //Each iteration generate one trajectory
    for _ in 0..n_trajectories {
        //History of all the moments in which something changed
        let mut time: Vec<f64> = Vec::new();
        //Configuration of the process variables at time t initialized with an uniform
        //distribution.
        let mut events: Vec<process::NetworkProcessState> = Vec::new();

        //Current Time and Current State
        let mut sample = sampler.next().unwrap();
        //Generate new samples until ending time is reached.
        while sample.t < t_end {
            time.push(sample.t);
            events.push(sample.state);
            sample = sampler.next().unwrap();
        }

        let current_state = events.last().unwrap().clone();
        events.push(current_state);

        //Add t_end as last time.
        time.push(t_end.clone());

        //Add the sampled trajectory to trajectories.
        trajectories.push(Trajectory::new(
            Array::from_vec(time),
            Array2::from_shape_vec(
                (events.len(), events.last().unwrap().len()),
                events
                    .iter()
                    .flatten()
                    .map(|x| match x {
                        params::StateType::Discrete(x) => x.clone(),
                    })
                    .collect(),
            )
            .unwrap(),
        ));
        sampler.reset();
    }
    //Return a dataset object with the sampled trajectories.
    Dataset::new(trajectories)
}

pub trait RandomGraphGenerator {
    fn new(density: f64, seed: Option<u64>) -> Self;
    fn generate_graph<T: NetworkProcess>(&mut self, net: &mut T);
}

/// Graph Generator using an uniform distribution.
///
/// A method to generate a random graph with edges uniformly distributed.
///
/// # Arguments
///
/// * `density` - is the density of the graph in terms of edges; domain: `0.0 ≤ density ≤ 1.0`.
/// * `rng` - is the random numbers generator.
///
/// # Example
///
/// ```rust
/// # use std::collections::BTreeSet;
/// # use ndarray::{arr1, arr2, arr3};
/// # use reCTBN::params;
/// # use reCTBN::params::Params::DiscreteStatesContinousTime;
/// # use reCTBN::tools::trajectory_generator;
/// # use reCTBN::process::NetworkProcess;
/// # use reCTBN::process::ctbn::CtbnNetwork;
/// use reCTBN::tools::UniformGraphGenerator;
/// use reCTBN::tools::RandomGraphGenerator;
/// # let mut net = CtbnNetwork::new();
/// # let nodes_cardinality = 8;
/// # let domain_cardinality = 4;
/// # for node in 0..nodes_cardinality {
/// #   // Create the domain for a discrete node
/// #   let mut domain = BTreeSet::new();
/// #   for dvalue in 0..domain_cardinality {
/// #     domain.insert(dvalue.to_string());
/// #   }
/// #   // Create the parameters for a discrete node using the domain
/// #   let param = params::DiscreteStatesContinousTimeParams::new(
/// #     node.to_string(),
/// #     domain
/// #   );
/// #   //Create the node using the parameters
/// #   let node = DiscreteStatesContinousTime(param);
/// #   // Add the node to the network
/// #   net.add_node(node).unwrap();
/// # }
///
/// // Initialize the Graph Generator using the one with an
/// // uniform distribution
/// let density = 1.0/3.0;
/// let seed = Some(7641630759785120);
/// let mut structure_generator = UniformGraphGenerator::new(
///     density,
///     seed
/// );
///
/// // Generate the graph directly on the network
/// structure_generator.generate_graph(&mut net);
/// # // Count all the edges generated in the network
/// # let mut edges = 0;
/// # for node in net.get_node_indices(){
/// #     edges += net.get_children_set(node).len()
/// # }
/// # // Number of all the nodes in the network
/// # let nodes = net.get_node_indices().len() as f64;
/// # let expected_edges = (density * nodes * (nodes - 1.0)).round() as usize;
/// # // ±10% of tolerance
/// # let tolerance = ((expected_edges as f64)*0.10) as usize;
/// # // As the way `generate_graph()` is implemented we can only reasonably
/// # // expect the number of edges to be somewhere around the expected value.
/// # assert!((expected_edges - tolerance) <= edges && edges <= (expected_edges + tolerance));
/// ```
pub struct UniformGraphGenerator {
    density: f64,
    rng: ChaCha8Rng,
}

impl RandomGraphGenerator for UniformGraphGenerator {
    fn new(density: f64, seed: Option<u64>) -> UniformGraphGenerator {
        if density < 0.0 || density > 1.0 {
            panic!(
                "Density value must be between 1.0 and 0.0, got {}.",
                density
            );
        }
        let rng: ChaCha8Rng = match seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };
        UniformGraphGenerator { density, rng }
    }

    /// Generate an uniformly distributed graph.
    fn generate_graph<T: NetworkProcess>(&mut self, net: &mut T) {
        net.initialize_adj_matrix();
        let last_node_idx = net.get_node_indices().len();
        for parent in 0..last_node_idx {
            for child in 0..last_node_idx {
                if parent != child {
                    if self.rng.gen_bool(self.density) {
                        net.add_edge(parent, child);
                    }
                }
            }
        }
    }
}

pub trait RandomParametersGenerator {
    fn new(interval: Range<f64>, seed: Option<u64>) -> Self;
    fn generate_parameters<T: NetworkProcess>(&mut self, net: &mut T);
}

/// Parameters Generator using an uniform distribution.
///
/// A method to generate random parameters uniformly distributed.
///
/// # Arguments
///
/// * `interval` - is the interval of the random values oh the CIM's diagonal; domain: `≥ 0.0`.
/// * `rng` - is the random numbers generator.
///
/// # Example
///
/// ```rust
/// # use std::collections::BTreeSet;
/// # use ndarray::{arr1, arr2, arr3};
/// # use reCTBN::params;
/// # use reCTBN::params::ParamsTrait;
/// # use reCTBN::params::Params::DiscreteStatesContinousTime;
/// # use reCTBN::process::NetworkProcess;
/// # use reCTBN::process::ctbn::CtbnNetwork;
/// # use reCTBN::tools::trajectory_generator;
/// # use reCTBN::tools::RandomGraphGenerator;
/// # use reCTBN::tools::UniformGraphGenerator;
/// use reCTBN::tools::RandomParametersGenerator;
/// use reCTBN::tools::UniformParametersGenerator;
/// # let mut net = CtbnNetwork::new();
/// # let nodes_cardinality = 8;
/// # let domain_cardinality = 4;
/// # for node in 0..nodes_cardinality {
/// #   // Create the domain for a discrete node
/// #   let mut domain = BTreeSet::new();
/// #   for dvalue in 0..domain_cardinality {
/// #     domain.insert(dvalue.to_string());
/// #   }
/// #   // Create the parameters for a discrete node using the domain
/// #   let param = params::DiscreteStatesContinousTimeParams::new(
/// #     node.to_string(),
/// #     domain
/// #   );
/// #   //Create the node using the parameters
/// #   let node = DiscreteStatesContinousTime(param);
/// #   // Add the node to the network
/// #   net.add_node(node).unwrap();
/// # }
/// #
/// # // Initialize the Graph Generator using the one with an
/// # // uniform distribution
/// # let mut structure_generator = UniformGraphGenerator::new(
/// #     1.0/3.0,
/// #     Some(7641630759785120)
/// # );
/// #
/// # // Generate the graph directly on the network
/// # structure_generator.generate_graph(&mut net);
///
/// // Initialize the parameters generator with uniform distributin
/// let mut cim_generator = UniformParametersGenerator::new(
///     0.0..7.0,
///     Some(7641630759785120)
/// );
///
/// // Generate CIMs with uniformly distributed parameters.
/// cim_generator.generate_parameters(&mut net);
/// #
/// # for node in net.get_node_indices() {
/// #     assert_eq!(
/// #         Ok(()),
/// #         net.get_node(node).validate_params()
/// #     );
/// }
/// ```
pub struct UniformParametersGenerator {
    interval: Range<f64>,
    rng: ChaCha8Rng,
}

impl RandomParametersGenerator for UniformParametersGenerator {
    fn new(interval: Range<f64>, seed: Option<u64>) -> UniformParametersGenerator {
        if interval.start < 0.0 || interval.end < 0.0 {
            panic!(
                "Interval must be entirely less or equal than 0, got {}..{}.",
                interval.start, interval.end
            );
        }
        let rng: ChaCha8Rng = match seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };
        UniformParametersGenerator { interval, rng }
    }

    /// Generate CIMs with uniformly distributed parameters.
    fn generate_parameters<T: NetworkProcess>(&mut self, net: &mut T) {
        for node in net.get_node_indices() {
            let parent_set_state_space_cardinality: usize = net
                .get_parent_set(node)
                .iter()
                .map(|x| net.get_node(*x).get_reserved_space_as_parent())
                .product();
            let node_domain_cardinality = net.get_node(node).get_reserved_space_as_parent();
            let mut cim = Array3::<f64>::from_shape_fn(
                (
                    parent_set_state_space_cardinality,
                    node_domain_cardinality,
                    node_domain_cardinality,
                ),
                |_| self.rng.gen(),
            );
            cim.axis_iter_mut(Axis(0)).for_each(|mut x| {
                x.diag_mut().fill(0.0);
                x.div_assign(&x.sum_axis(Axis(1)).insert_axis(Axis(1)));
                let diag = Array1::<f64>::from_shape_fn(node_domain_cardinality, |_| {
                    self.rng.gen_range(self.interval.clone())
                });
                x.mul_assign(&diag.clone().insert_axis(Axis(1)));
                // Recomputing the diagonal in order to reduce the issues caused by the loss of
                // precision when validating the parameters.
                let diag_sum = -x.sum_axis(Axis(1));
                x.diag_mut().assign(&diag_sum)
            });
            match &mut net.get_node_mut(node) {
                params::Params::DiscreteStatesContinousTime(param) => {
                    param.set_cim_unchecked(cim);
                }
            }
        }
    }
}
