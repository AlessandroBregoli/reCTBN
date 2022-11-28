pub mod reward_function;

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

pub trait RewardFunction {
    /// Given the current state and the previous state, it compute the reward.
    ///
    /// # Arguments
    ///
    /// * `current_state`: the current state of the network represented as a `process::NetworkProcessState`
    /// * `previous_state`: an optional argument representing the previous state of the network

    fn call(
        &self,
        current_state: process::NetworkProcessState,
        previous_state: Option<process::NetworkProcessState>,
    ) -> Reward;

    /// Initialize the RewardFunction internal accordingly to the structure of a NetworkProcess
    ///
    /// # Arguments
    ///
    /// * `p`: any structure that implements the trait `process::NetworkProcess`
    fn initialize_from_network_process<T: process::NetworkProcess>(p: &T) -> Self;
}
