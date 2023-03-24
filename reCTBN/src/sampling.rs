//! Module containing methods for the sampling.

use crate::{
    params::ParamsTrait,
    process::{NetworkProcess, NetworkProcessState},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// This structure represent one `sample` of a trajectory.
///
/// # Attributes
///
/// * `t` - time instant of the sample
/// * `state` - state of the `NetworkProcess` in the time instant `t`
#[derive(Clone)]
pub struct Sample {
    pub t: f64,
    pub state: NetworkProcessState,
}

/// The `trait Sampler` is an iterator that generate a sequence of `Sample`. 
pub trait Sampler: Iterator<Item = Sample> {

    /// Reset the Sampler to the initial state.
    fn reset(&mut self);
}

/// This structure implements the `Sampler` and allow to generate a sequence of `Sample`
/// accordingly to *(Fan, Yu, and Christian R. Shelton. "Sampling for Approximate Inference in 
/// Continuous Time Bayesian Networks." ISAIM. 2008.)*
///
///  # Attributes
///
///  * `net` - a structure implementing the `trait NetworkProcess`
///  * `rng` - a random number generator
///  * `current_time` - current time of the sampler. This variable will be update every time the
///                    sampler generate a sample
///  * `current_state` - current state of the underline `NetworkProcess`. This variable will be
///                     update every time the sampler generate a sample  
///  * `next_transitions` - next time to transition for each variable in the
///                       `NetworkProcess`
///  * `initial_state`: - Initial state of the `NetworkProcess`
///
///  # Example
///
///```rust
/// use reCTBN::params;
/// use reCTBN::process::{ctbn::CtbnNetwork, NetworkProcess, NetworkProcessState};
/// use reCTBN::sampling::{ForwardSampler, Sampler, Sample};
/// use ndarray::*;
/// use std::collections::BTreeSet;
///
/// //Create the domain for a discrete node
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
///
/// //Create the parameters for a discrete node using the domain
/// let param = params::DiscreteStatesContinousTimeParams::new("X1".to_string(), domain);
///
/// //Create the node using the parameters
/// let X1 = params::Params::DiscreteStatesContinousTime(param);
///
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
/// let param = params::DiscreteStatesContinousTimeParams::new("X2".to_string(), domain);
/// let X2 = params::Params::DiscreteStatesContinousTime(param);
///
/// //Initialize a ctbn
/// let mut net = CtbnNetwork::new();
///
/// //Add nodes
/// let X1 = net.add_node(X1).unwrap();
/// let X2 = net.add_node(X2).unwrap();
///
/// //Add an edge
/// net.add_edge(X1, X2);
///
///  //Initialize the cims
///
///  match &mut net.get_node_mut(X1) {
///      params::Params::DiscreteStatesContinousTime(param) => {
///          assert_eq!(Ok(()), param.set_cim(arr3(&[[[-0.1, 0.1], [1.0, -1.0]]])));
///      }
///  }
///
///  match &mut net.get_node_mut(X2) {
///      params::Params::DiscreteStatesContinousTime(param) => {
///          assert_eq!(
///              Ok(()),
///              param.set_cim(arr3(&[
///                  [[-0.01, 0.01], [5.0, -5.0]],
///                  [[-5.0, 5.0], [0.01, -0.01]]
///              ]))
///          );
///      }
///  }
///
/// // Define an initial state for the ctbn (X1 = 0, X2 = 0)
/// let s0: NetworkProcessState = vec![params::StateType::Discrete(0), params::StateType::Discrete(0)];
///
///
/// //initialize the Forward Sampler
///
///  let mut sampler = ForwardSampler::new(&net, Some(1994), Some(s0.clone()));
///
///  //The first output of the iterator will be t=0 and state=s0
///  let sample_t0 = sampler.next().unwrap();
///  assert_eq!(0.0, sample_t0.t);
///  assert_eq!(s0, sample_t0.state);
/// 
///```
pub struct ForwardSampler<'a, T>
where
    T: NetworkProcess,
{
    net: &'a T,
    rng: ChaCha8Rng,
    current_time: f64,
    current_state: NetworkProcessState,
    next_transitions: Vec<Option<f64>>,
    initial_state: Option<NetworkProcessState>,
}

impl<'a, T: NetworkProcess> ForwardSampler<'a, T> {

    /// Constructur method for `ForwardSampler`
    ///
    /// # Arguments
    ///
    /// * `net` - A structure implementing the `NetworkProcess` trait
    /// * `seed` - Random seed used to make the trajectory generation reproducible
    /// * `initial_state` - Initial state of the `NetworkProcess`. If none, an initial state will be 
    ///    sampled
    pub fn new(
        net: &'a T,
        seed: Option<u64>,
        initial_state: Option<NetworkProcessState>,
    ) -> ForwardSampler<'a, T> {
        let rng: ChaCha8Rng = match seed {
            //If a seed is present use it to initialize the random generator.
            Some(seed) => SeedableRng::seed_from_u64(seed),
            //Otherwise create a new random generator using the method `from_entropy`
            None => SeedableRng::from_entropy(),
        };
        let mut fs = ForwardSampler {
            net,
            rng,
            current_time: 0.0,
            current_state: vec![],
            next_transitions: vec![],
            initial_state,
        };
        fs.reset();
        return fs;
    }
}

impl<'a, T: NetworkProcess> Iterator for ForwardSampler<'a, T> {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        // Set the variable to be returned (time and state)
        let ret_time = self.current_time.clone();
        let ret_state = self.current_state.clone();
        
        //  All the operation stating from here are required to compute the time and state that
        //  will be returned at the next call of this function. 
        
        //Check if there are any node without a next time to transition and sample it from an
        //exponential distribution governed by the main diagonal of the CIM.
        for (idx, val) in self.next_transitions.iter_mut().enumerate() {
            if let None = val {
                *val = Some(
                    self.net
                        .get_node(idx)
                        .get_random_residence_time(
                            self.net
                                .get_node(idx)
                                .state_to_index(&self.current_state[idx]),
                            self.net.get_param_index_network(idx, &self.current_state),
                            &mut self.rng,
                        )
                        .unwrap()
                        + self.current_time,
                );
            }
        }
        
        //The next node to transition will be the node with the smallest value in next_transitions
        let next_node_transition = self
            .next_transitions
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.unwrap().partial_cmp(&y.1.unwrap()).unwrap())
            .unwrap()
            .0;

        self.current_time = self.next_transitions[next_node_transition].unwrap().clone();
        
        // Generate the new  state of the node from a multinomial distribution governed by the off
        // diagonal parameters of the CIM.
        self.current_state[next_node_transition] = self
            .net
            .get_node(next_node_transition)
            .get_random_state(
                self.net
                    .get_node(next_node_transition)
                    .state_to_index(&self.current_state[next_node_transition]),
                self.net
                    .get_param_index_network(next_node_transition, &self.current_state),
                &mut self.rng,
            )
            .unwrap();

        //Reset the next_transition for the transitioning node.
        self.next_transitions[next_node_transition] = None;

        //Reset the next_transition for each child of the transitioning node.
        for child in self.net.get_children_set(next_node_transition) {
            self.next_transitions[child] = None;
        }

        Some(Sample {
            t: ret_time,
            state: ret_state,
        })
    }
}

impl<'a, T: NetworkProcess> Sampler for ForwardSampler<'a, T> {
    fn reset(&mut self) {
        self.current_time = 0.0;
        match &self.initial_state {
            None => {
                self.current_state = self
                    .net
                    .get_node_indices()
                    .map(|x| self.net.get_node(x).get_random_state_uniform(&mut self.rng))
                    .collect()
            }
            Some(is) => self.current_state = is.clone(),
        };
        self.next_transitions = self.net.get_node_indices().map(|_| Option::None).collect();
    }
}
