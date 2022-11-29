use crate::{
    reward::RewardEvaluation,
    sampling::{ForwardSampler, Sampler},
    process::NetworkProcessState
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
    fn call<N: crate::process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
    ) -> ndarray::Array1<f64> {
        todo!()
    }

    fn call_state<N: crate::process::NetworkProcess, R: super::RewardFunction>(
        &self,
        network_process: &N,
        reward_function: &R,
        state: &NetworkProcessState,
    ) -> f64 {
        let mut sampler = ForwardSampler::new(network_process, self.seed.clone(), Some(state.clone()));
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
                    ret += std::f64::consts::E.powf(-self.discount_factor * current.t) * r.transition_reward;
                }
                previous = current;
            }
        }

        ret / self.n_iterations as f64
    }
}
