//! Module for dealing with reward functions

use crate::{
    params::{self, ParamsTrait},
    process, sampling,
};
use ndarray;

/// Instantiation of reward function and instantaneous reward
///
///
/// # Arguments
///
/// * `transition_reward`: reward obtained transitioning from one state to another
/// * `instantaneous_reward`: reward per unit of time obtained staying in a specific state

#[derive(Debug, PartialEq)]
pub struct Reward {
    pub transition_reward: f64,
    pub instantaneous_reward: f64,
}

/// The trait RewardFunction describe the methods that all the reward functions must satisfy

pub trait RewardFunction {
    /// Given the current state and the previous state, it compute the reward.
    ///
    /// # Arguments
    ///
    /// * `current_state`: the current state of the network represented as a `sampling::Sample`
    /// * `previous_state`: an optional argument representing the previous state of the network

    fn call(
        &self,
        current_state: sampling::Sample,
        previous_state: Option<sampling::Sample>,
    ) -> Reward;

    /// Initialize the RewardFunction internal accordingly to the structure of a NetworkProcess
    ///
    /// # Arguments
    ///
    /// * `p`: any structure that implements the trait `process::NetworkProcess`
    fn initialize_from_network_process<T: process::NetworkProcess>(p: &T) -> Self;
}

/// Reward function over a factored state space
///
/// The `FactoredRewardFunction` assume the reward function is the sum of the reward of each node
/// of the underling `NetworkProcess`
///
/// # Arguments
///
/// * `transition_reward`: a vector of two-dimensional arrays. Each array contains the transition
/// reward of a node

pub struct FactoredRewardFunction {
    transition_reward: Vec<ndarray::Array2<f64>>,
    instantaneous_reward: Vec<ndarray::Array1<f64>>,
}

impl FactoredRewardFunction {
    pub fn get_transition_reward(&self, node_idx: usize) -> &ndarray::Array2<f64> {
        &self.transition_reward[node_idx]
    }

    pub fn get_transition_reward_mut(&mut self, node_idx: usize) -> &mut ndarray::Array2<f64> {
        &mut self.transition_reward[node_idx]
    }

    pub fn get_instantaneous_reward(&self, node_idx: usize) -> &ndarray::Array1<f64> {
        &self.instantaneous_reward[node_idx]
    }

    pub fn get_instantaneous_reward_mut(&mut self, node_idx: usize) -> &mut ndarray::Array1<f64> {
        &mut self.instantaneous_reward[node_idx]
    }
}

impl RewardFunction for FactoredRewardFunction {
    fn call(
        &self,
        current_state: sampling::Sample,
        previous_state: Option<sampling::Sample>,
    ) -> Reward {
        let instantaneous_reward: f64 = current_state
            .state
            .iter()
            .enumerate()
            .map(|(idx, x)| {
                let x = match x {
                    params::StateType::Discrete(x) => x,
                };
                self.instantaneous_reward[idx][*x]
            })
            .sum();
        if let Some(previous_state) = previous_state {
            let transition_reward = previous_state
                .state
                .iter()
                .zip(current_state.state.iter())
                .enumerate()
                .find_map(|(idx, (p, c))| -> Option<f64> {
                    let p = match p {
                        params::StateType::Discrete(p) => p,
                    };
                    let c = match c {
                        params::StateType::Discrete(c) => c,
                    };
                    if p != c {
                        Some(self.transition_reward[idx][[*p, *c]])
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0);
            Reward {
                transition_reward,
                instantaneous_reward,
            }
        } else {
            Reward {
                transition_reward: 0.0,
                instantaneous_reward,
            }
        }
    }

    fn initialize_from_network_process<T: process::NetworkProcess>(p: &T) -> Self {
        let mut transition_reward: Vec<ndarray::Array2<f64>> = vec![];
        let mut instantaneous_reward: Vec<ndarray::Array1<f64>> = vec![];
        for i in p.get_node_indices() {
            //This works only for discrete nodes!
            let size: usize = p.get_node(i).get_reserved_space_as_parent();
            instantaneous_reward.push(ndarray::Array1::zeros(size));
            transition_reward.push(ndarray::Array2::zeros((size, size)));
        }

        FactoredRewardFunction {
            transition_reward,
            instantaneous_reward,
        }
    }
}
