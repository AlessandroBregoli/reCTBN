/// Evaluate the `RewardFunction` for a `NetworkProcess`

use std::collections::HashMap;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use statrs::distribution::ContinuousCDF;

use crate::params::{self, ParamsTrait};
use crate::process;

use crate::{
    process::NetworkProcessState,
    reward::RewardEvaluation,
    sampling::{ForwardSampler, Sampler},
};

/// Supported types of `RewardCriteria`
///
/// # Variants
///
/// * `RewardCriteria::FiniteHorizon` - reward over a finite horizon
/// * `RewardCriteria::InfiniteHorizon { discount_factor: f64}` - 
///     discounted reward over an infinite horizon
pub enum RewardCriteria {
    FiniteHorizon,
    InfiniteHorizon { discount_factor: f64 },
}


/// Monte Carlo algorithm to approximate the evaluation of the reward function
/// 
/// # Arguments
/// 
/// * `max_iteration`: maximum number of iteration (number of trajectory generated)
/// * `max_err_stop`: maximum absolute error used of the early stopping rule
/// * `alpha_stop`: alpha used by the early stopping rule 
/// * `end_time`: ending time used for the generation of each trajectory
/// * `reward_criteria`: Reward criteria used for evaluate the reward function
/// * `seed`: Seed used by the random generator
///
/// # Example
///
///  ```rust
/// 
/// use approx::assert_abs_diff_eq;
/// use ndarray::*;
/// use reCTBN::{
///     params,
///     process::{ctbn::*, NetworkProcess, NetworkProcessState},
///     reward::{reward_evaluation::*, reward_function::*, *},
/// };
/// use std::collections::BTreeSet;
///
/// //Create the domain for a discrete node
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
///
/// //Create the parameters for a discrete node using the domain
/// let param = params::DiscreteStatesContinousTimeParams::new("n1".to_string(), domain);
///
/// //Create the node using the parameters
/// let n1 = params::Params::DiscreteStatesContinousTime(param);
///
/// // Initialize the CTBN
/// let mut net = CtbnNetwork::new();
///
/// // Add the node n1 to the network
/// let n1 = net
///     .add_node(n1)
///     .unwrap();
/// 
/// // Initialize the reward based no `n1`
/// let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
/// rf.get_transition_reward_mut(n1)
///     .assign(&arr2(&[[0.0, 0.0], [0.0, 0.0]]));
/// rf.get_instantaneous_reward_mut(n1)
///     .assign(&arr1(&[3.0, 3.0]));
///
/// //Set the CIM for n1
/// match &mut net.get_node_mut(n1) {
///     params::Params::DiscreteStatesContinousTime(param) => {
///         param.set_cim(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]])).unwrap();
///     }
/// }
///
/// net.initialize_adj_matrix();
/// 
/// // Define the possible states for the network
/// let s0: NetworkProcessState = vec![params::StateType::Discrete(0)];
/// let s1: NetworkProcessState = vec![params::StateType::Discrete(1)];
///
/// //Initialize the `MonteCarloReward` with an infinite reward criteria
/// let mc = MonteCarloReward::new(10000, 1e-1, 1e-1, 10.0, RewardCriteria::InfiniteHorizon { discount_factor: 1.0 }, Some(215));
///
/// let rst = mc.evaluate_state_space(&net, &rf);
/// assert_abs_diff_eq!(3.0, rst[&s0], epsilon = 1e-2);
/// assert_abs_diff_eq!(3.0, rst[&s1], epsilon = 1e-2);
///
///
/// let mc = MonteCarloReward::new(10000, 1e-1, 1e-1, 10.0, RewardCriteria::FiniteHorizon, Some(215));
/// assert_abs_diff_eq!(30.0, mc.evaluate_state(&net, &rf, &s0), epsilon = 1e-2);
/// assert_abs_diff_eq!(30.0, mc.evaluate_state(&net, &rf, &s1), epsilon = 1e-2);
/// ```
pub struct MonteCarloReward {
    max_iterations: usize,
    max_err_stop: f64,
    alpha_stop: f64,
    end_time: f64,
    reward_criteria: RewardCriteria,
    seed: Option<u64>,
}

impl MonteCarloReward {
    pub fn new(
        max_iterations: usize,
        max_err_stop: f64,
        alpha_stop: f64,
        end_time: f64,
        reward_criteria: RewardCriteria,
        seed: Option<u64>,
    ) -> MonteCarloReward {
        MonteCarloReward {
            max_iterations,
            max_err_stop,
            alpha_stop,
            end_time,
            reward_criteria,
            seed,
        }
    }
}

impl RewardEvaluation for MonteCarloReward {
    fn evaluate_state_space<N: process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
    ) -> HashMap<process::NetworkProcessState, f64> {
        // Domain size of each variable in the `NetworkProcess`
        let variables_domain: Vec<Vec<params::StateType>> = network_process
            .get_node_indices()
            .map(|x| match network_process.get_node(x) {
                params::Params::DiscreteStatesContinousTime(x) => (0..x
                    .get_reserved_space_as_parent())
                    .map(|s| params::StateType::Discrete(s))
                    .collect(),
            })
            .collect();
        
        // Number of possible configuration of the `NetworkProcess`
        let n_states: usize = variables_domain.iter().map(|x| x.len()).product();
        
        // Compute the expected reward for each possible configuration of the `NetworkProcess`
        (0..n_states)
            .into_par_iter()
            .map(|s| {
                let state: process::NetworkProcessState = variables_domain
                    .iter()
                    .fold((s, vec![]), |acc, x| {
                        let mut acc = acc;
                        let idx_s = acc.0 % x.len();
                        acc.1.push(x[idx_s].clone());
                        acc.0 = acc.0 / x.len();
                        acc
                    })
                    .1;

                let r = self.evaluate_state(network_process, reward_function, &state);
                (state, r)
            })
            .collect()
    }

    fn evaluate_state<N: crate::process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
        state: &NetworkProcessState,
    ) -> f64 {
        // Initialize the Forward Sampler.
        let mut sampler =
            ForwardSampler::new(network_process, self.seed.clone(), Some(state.clone()));

        // Initialize the variable required to perform early stopping hypotesis test
        let mut expected_value = 0.0;
        let mut squared_expected_value = 0.0;
        let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
        
        // Generate and evaluate tranjectories util max_iteration is reached or early stopping rule
        // is satisfied.
        for i in 0..self.max_iterations {
            // Reset the sampler (Set time to 0 and initial value to `state`)
            sampler.reset();
            let mut ret = 0.0;
            let mut previous = sampler.next().unwrap();

            // Generate transitions until `end_time` is reached
            while previous.t < self.end_time {
                let current = sampler.next().unwrap();
                if current.t > self.end_time {
                    let r = reward_function.call(&previous.state, None);
                    let discount = match self.reward_criteria {
                        RewardCriteria::FiniteHorizon => self.end_time - previous.t,
                        RewardCriteria::InfiniteHorizon { discount_factor } => {
                            std::f64::consts::E.powf(-discount_factor * previous.t)
                                - std::f64::consts::E.powf(-discount_factor * self.end_time)
                        }
                    };
                    ret += discount * r.instantaneous_reward;
                } else {
                    let r = reward_function.call(&current.state, Some(&previous.state));
                    let discount = match self.reward_criteria {
                        RewardCriteria::FiniteHorizon => current.t - previous.t,
                        RewardCriteria::InfiniteHorizon { discount_factor } => {
                            std::f64::consts::E.powf(-discount_factor * previous.t)
                                - std::f64::consts::E.powf(-discount_factor * current.t)
                        }
                    };
                    ret += discount * r.instantaneous_reward;
                    ret += match self.reward_criteria {
                        RewardCriteria::FiniteHorizon => 1.0,
                        RewardCriteria::InfiniteHorizon { discount_factor } => {
                            std::f64::consts::E.powf(-discount_factor * current.t)
                        }
                    } * r.transition_reward;
                }
                previous = current;
            }
            
            // Evaluate the early stopping hypothesis test .
            let float_i = i as f64;
            expected_value =
                expected_value * float_i as f64 / (float_i + 1.0) + ret / (float_i + 1.0);
            squared_expected_value = squared_expected_value * float_i as f64 / (float_i + 1.0)
                + ret.powi(2) / (float_i + 1.0);

            if i > 2 {
                let var =
                    (float_i + 1.0) / float_i * (squared_expected_value - expected_value.powi(2));
                if self.alpha_stop
                    - 2.0 * normal.cdf(-(float_i + 1.0).sqrt() * self.max_err_stop / var.sqrt())
                    > 0.0
                {
                    return expected_value;
                }
            }
        }

        expected_value
    }
}


/// Compute the Neighborhood Relative Reward
///
/// The Neighborhood Relative Reward is the maximum ratio between the expected reward of the
/// current state and the expected reward of each state reachable  with one transition.
/// 
/// # Arguments
///
/// *  `inner_reward`: a structure implementing the `trait RewardEvaluation`
///
/// # Example
///
///  ```rust
/// 
/// use approx::assert_abs_diff_eq;
/// use ndarray::*;
/// use reCTBN::{
///     params,
///     process::{ctbn::*, NetworkProcess, NetworkProcessState},
///     reward::{reward_evaluation::*, reward_function::*, *},
/// };
/// use std::collections::BTreeSet;
///
/// //Create the domain for a discrete node
/// let mut domain = BTreeSet::new();
/// domain.insert(String::from("A"));
/// domain.insert(String::from("B"));
///
/// //Create the parameters for a discrete node using the domain
/// let param = params::DiscreteStatesContinousTimeParams::new("n1".to_string(), domain);
///
/// //Create the node using the parameters
/// let n1 = params::Params::DiscreteStatesContinousTime(param);
///
/// // Initialize the CTBN
/// let mut net = CtbnNetwork::new();
///
/// // Add the node n1 to the network
/// let n1 = net
///     .add_node(n1)
///     .unwrap();
/// 
/// // Initialize the reward based no `n1`
/// let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
/// rf.get_transition_reward_mut(n1)
///     .assign(&arr2(&[[0.0, 0.0], [0.0, 0.0]]));
/// rf.get_instantaneous_reward_mut(n1)
///     .assign(&arr1(&[3.0, 3.0]));
///
/// //Set the CIM for n1
/// match &mut net.get_node_mut(n1) {
///     params::Params::DiscreteStatesContinousTime(param) => {
///         param.set_cim(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]])).unwrap();
///     }
/// }
///
/// net.initialize_adj_matrix();
/// 
/// // Define the possible states for the network
/// let s0: NetworkProcessState = vec![params::StateType::Discrete(0)];
/// let s1: NetworkProcessState = vec![params::StateType::Discrete(1)];
///
/// //Initialize the `MonteCarloReward` with an infinite reward criteria
/// let mc = MonteCarloReward::new(10000, 1e-1, 1e-1, 100.0, RewardCriteria::InfiniteHorizon { discount_factor: 0.1 }, Some(215));
///
/// let nrr = NeighborhoodRelativeReward::new(mc);
///
/// let rst = nrr.evaluate_state_space(&net, &rf);
/// assert_abs_diff_eq!(1.0, rst[&s0], epsilon = 1e-2);
/// assert_abs_diff_eq!(1.0, rst[&s1], epsilon = 1e-2);
/// ```
pub struct NeighborhoodRelativeReward<RE: RewardEvaluation> {
    inner_reward: RE,
}

impl<RE: RewardEvaluation> NeighborhoodRelativeReward<RE> {
    pub fn new(inner_reward: RE) -> NeighborhoodRelativeReward<RE> {
        NeighborhoodRelativeReward { inner_reward }
    }
}

impl<RE: RewardEvaluation> RewardEvaluation for NeighborhoodRelativeReward<RE> {
    fn evaluate_state_space<N: process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
    ) -> HashMap<process::NetworkProcessState, f64> {
        let absolute_reward = self
            .inner_reward
            .evaluate_state_space(network_process, reward_function);

        //This approach optimize memory. Maybe optimizing execution time can be better.
        absolute_reward
            .iter()
            .map(|(k1, v1)| {
                let mut max_val: f64 = 1.0;
                absolute_reward.iter().for_each(|(k2, v2)| {
                    let count_diff: usize = k1
                        .iter()
                        .zip(k2.iter())
                        .map(|(s1, s2)| if s1 == s2 { 0 } else { 1 })
                        .sum();
                    if count_diff < 2 {
                        max_val = max_val.max(v1 / v2);
                    }
                });
                (k1.clone(), max_val)
            })
            .collect()
    }

    fn evaluate_state<N: process::NetworkProcess, R: super::RewardFunction>(
        &self,
        _network_process: &N,
        _reward_function: &R,
        _state: &process::NetworkProcessState,
    ) -> f64 {
        unimplemented!();
    }
}
