//! Module containing methods for the sampling.

use crate::{
    params::ParamsTrait,
    process::{NetworkProcess, NetworkProcessState},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
pub struct Sample {
    pub t: f64,
    pub state: NetworkProcessState,
}

pub trait Sampler: Iterator<Item = Sample> {
    fn reset(&mut self);
}

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
        let ret_time = self.current_time.clone();
        let ret_state = self.current_state.clone();

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

        let next_node_transition = self
            .next_transitions
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.unwrap().partial_cmp(&y.1.unwrap()).unwrap())
            .unwrap()
            .0;

        self.current_time = self.next_transitions[next_node_transition].unwrap().clone();

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

        self.next_transitions[next_node_transition] = None;

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
