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
                x.diag_mut().assign(&-diag)
            });
            match &mut net.get_node_mut(node) {
                params::Params::DiscreteStatesContinousTime(param) => {
                    param.set_cim_unchecked(cim);
                }
            }
        }
    }
}
