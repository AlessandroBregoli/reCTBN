use std::collections::HashMap;

use crate::params::{self, ParamsTrait};
use crate::process;

use crate::{
    process::NetworkProcessState,
    reward::RewardEvaluation,
    sampling::{ForwardSampler, Sampler},
};

pub enum RewardCriteria {
    FiniteHorizon,
    InfiniteHorizon {discount_factor: f64},
}

pub struct MonteCarloReward {
    n_iterations: usize,
    end_time: f64,
    reward_criteria: RewardCriteria,
    seed: Option<u64>,
}

impl MonteCarloReward {
    pub fn new(
        n_iterations: usize,
        end_time: f64,
        reward_criteria: RewardCriteria,
        seed: Option<u64>,
    ) -> MonteCarloReward {
        MonteCarloReward {
            n_iterations,
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
        let variables_domain: Vec<Vec<params::StateType>> = network_process
            .get_node_indices()
            .map(|x| match network_process.get_node(x) {
                params::Params::DiscreteStatesContinousTime(x) => (0..x
                    .get_reserved_space_as_parent())
                    .map(|s| params::StateType::Discrete(s))
                    .collect(),
            })
            .collect();

        let n_states: usize = variables_domain.iter().map(|x| x.len()).product();

        (0..n_states)
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
        let mut sampler =
            ForwardSampler::new(network_process, self.seed.clone(), Some(state.clone()));
        let mut ret = 0.0;

        for _i in 0..self.n_iterations {
            sampler.reset();
            let mut previous = sampler.next().unwrap();
            while previous.t < self.end_time {
                let current = sampler.next().unwrap();
                if current.t > self.end_time {
                    let r = reward_function.call(&previous.state, None);
                    let discount = match self.reward_criteria {
                        RewardCriteria::FiniteHorizon => self.end_time - previous.t,
                        RewardCriteria::InfiniteHorizon {discount_factor} => {
                            std::f64::consts::E.powf(-discount_factor * previous.t)
                                - std::f64::consts::E.powf(-discount_factor * self.end_time)
                        }
                    };
                    ret += discount * r.instantaneous_reward;
                } else {
                    let r = reward_function.call(&previous.state, Some(&current.state));
                    let discount = match self.reward_criteria {
                        RewardCriteria::FiniteHorizon => current.t-previous.t,
                        RewardCriteria::InfiniteHorizon {discount_factor} => {
                            std::f64::consts::E.powf(-discount_factor * previous.t)
                                - std::f64::consts::E.powf(-discount_factor * current.t)
                        }
                    };
                    ret += discount * r.instantaneous_reward;
                    ret += match self.reward_criteria {
                        RewardCriteria::FiniteHorizon => 1.0,
                        RewardCriteria::InfiniteHorizon {discount_factor} => {
                            std::f64::consts::E.powf(-discount_factor * current.t)
                        }
                    } * r.transition_reward;
                }
                previous = current;
            }
        }

        ret / self.n_iterations as f64
    }
}

pub struct NeighborhoodRelativeReward<RE: RewardEvaluation> {
    inner_reward: RE
}

impl<RE: RewardEvaluation> NeighborhoodRelativeReward<RE>{
    pub fn new(inner_reward: RE) -> NeighborhoodRelativeReward<RE>{
        NeighborhoodRelativeReward {inner_reward}
    }
}

impl<RE:RewardEvaluation> RewardEvaluation for NeighborhoodRelativeReward<RE> {
    fn evaluate_state_space<N: process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
    ) -> HashMap<process::NetworkProcessState, f64> {

        let absolute_reward = self.inner_reward.evaluate_state_space(network_process, reward_function);
        
        //This approach optimize memory. Maybe optimizing execution time can be better.
        absolute_reward.iter().map(|(k1, v1)| {
            let mut max_val:f64 = 1.0;
            absolute_reward.iter().for_each(|(k2,v2)| {
                let count_diff:usize = k1.iter().zip(k2.iter()).map(|(s1, s2)| if s1 == s2 {0} else {1}).sum();
                if count_diff < 2 {
                    max_val = max_val.max(v1/v2);
                }
            
            });
            (k1.clone(), max_val)
        }).collect()
        
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
