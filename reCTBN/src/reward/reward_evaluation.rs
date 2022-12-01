use std::collections::HashMap;

use crate::params::{self, ParamsTrait};
use crate::process;

use crate::{
    process::NetworkProcessState,
    reward::RewardEvaluation,
    sampling::{ForwardSampler, Sampler},
};

pub struct MonteCarloDiscountedRward {
    n_iterations: usize,
    end_time: f64,
    discount_factor: f64,
    seed: Option<u64>,
}

impl MonteCarloDiscountedRward {
    pub fn new(
        n_iterations: usize,
        end_time: f64,
        discount_factor: f64,
        seed: Option<u64>,
    ) -> MonteCarloDiscountedRward {
        MonteCarloDiscountedRward {
            n_iterations,
            end_time,
            discount_factor,
            seed,
        }
    }
}

impl RewardEvaluation for MonteCarloDiscountedRward {
    fn evaluate_state_space<N: process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
    ) -> HashMap<process::NetworkProcessState, f64> {
        let variables_domain: Vec<Vec<params::StateType>> = network_process
            .get_node_indices()
            .map(|x| match network_process.get_node(x) {
                params::Params::DiscreteStatesContinousTime(x) => 
                    (0..x.get_reserved_space_as_parent()).map(|s| params::StateType::Discrete(s)).collect()
                }).collect();

        let n_states:usize = variables_domain.iter().map(|x| x.len()).product();
        
        (0..n_states).map(|s| {
            let state: process::NetworkProcessState = variables_domain.iter().fold((s, vec![]), |acc, x| {
                let mut acc = acc;
                let idx_s = acc.0%x.len();
                acc.1.push(x[idx_s].clone());
                acc.0 = acc.0 / x.len();
                acc
            }).1;

            let r = self.evaluate_state(network_process, reward_function, &state);
            (state, r)
        }).collect()
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
                    let discount = std::f64::consts::E.powf(-self.discount_factor * previous.t)
                        - std::f64::consts::E.powf(-self.discount_factor * self.end_time);
                    ret += discount * r.instantaneous_reward;
                } else {
                    let r = reward_function.call(&previous.state, Some(&current.state));
                    let discount = std::f64::consts::E.powf(-self.discount_factor * previous.t)
                        - std::f64::consts::E.powf(-self.discount_factor * current.t);
                    ret += discount * r.instantaneous_reward;
                    ret += std::f64::consts::E.powf(-self.discount_factor * current.t)
                        * r.transition_reward;
                }
                previous = current;
            }
        }

        ret / self.n_iterations as f64
    }
}
