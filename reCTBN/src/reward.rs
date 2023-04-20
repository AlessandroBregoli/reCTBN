pub mod reward_evaluation;
pub mod reward_function;

use std::collections::HashMap;

use crate::process;

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

pub trait RewardFunction: Sync {
    /// Given the current state and the previous state, it compute the reward.
    ///
    /// # Arguments
    ///
    /// * `current_state`: the current state of the network represented as a `process::NetworkProcessState`
    /// * `previous_state`: an optional argument representing the previous state of the network

    fn call(
        &self,
        current_state: &process::NetworkProcessState,
        previous_state: Option<&process::NetworkProcessState>,
    ) -> Reward;

    /// Initialize the RewardFunction internal accordingly to the structure of a NetworkProcess
    ///
    /// # Arguments
    ///
    /// * `p`: any structure that implements the trait `process::NetworkProcess`
    fn initialize_from_network_process<T: process::NetworkProcess>(p: &T) -> Self;
}

/// The trait RewardEvaluation descibe the methods that all reward evaluation functors must satisfy.
pub trait RewardEvaluation {
    /// Evaluate the reward_function for all the possible configurations
    ///
    /// # Arguments
    ///
    /// * `network_process`: a `NetworkProcess` instance.
    /// * `reward_function`: the reward functin used over the network_process
    ///
    /// # Return
    ///
    /// * Return the reward for all the possible configurations of `network_process`.
    fn evaluate_state_space<N: process::NetworkProcess, R: RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
    ) -> HashMap<process::NetworkProcessState, f64>;

    /// Evaluate the reward_function for a single state
    ///
    /// # Arguments
    ///
    /// * `network_process`: a `NetworkProcess` instance.
    /// * `reward_function`: the reward functin used over the network_process
    /// * `state`: specific configuration of the `network_process`.
    ///
    /// # Return
    ///
    /// * Return the reward for the specific instance as an `f64` value.
    fn evaluate_state<N: process::NetworkProcess, R: RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
        state: &process::NetworkProcessState,
    ) -> f64;
}
