use crate::network;
use crate::node;
use crate::params;
use crate::params::ParamsTrait;
use ndarray::prelude::*;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct Trajectory {
    time: Array1<f64>,
    events: Array2<usize>,
}

impl Trajectory {
    pub fn init(time: Array1<f64>, events: Array2<usize>) -> Trajectory {
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

pub struct Dataset {
    trajectories: Vec<Trajectory>,
}

impl Dataset {
    pub fn init(trajectories: Vec<Trajectory>) -> Dataset {
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

pub fn trajectory_generator<T: network::Network>(
    net: &T,
    n_trajectories: u64,
    t_end: f64,
    seed: Option<u64>,
) -> Dataset {

    let mut trajectories: Vec<Trajectory> = Vec::new();
    let seed = seed.unwrap_or_else(rand::random);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let node_idx: Vec<_> = net.get_node_indices().collect();
    for _ in 0..n_trajectories {
        let mut t = 0.0;
        let mut time: Vec<f64> = Vec::new();
        let mut events: Vec<Array1<usize>> = Vec::new();
        let mut current_state: Vec<params::StateType> = node_idx
            .iter()
            .map(|x| net.get_node(*x).params.get_random_state_uniform(&mut rng))
            .collect();
        let mut next_transitions: Vec<Option<f64>> =
            (0..node_idx.len()).map(|_| Option::None).collect();
        events.push(
            current_state
                .iter()
                .map(|x| match x {
                    params::StateType::Discrete(state) => state.clone(),
                })
                .collect(),
        );
        time.push(t.clone());
        while t < t_end {
            for (idx, val) in next_transitions.iter_mut().enumerate() {
                if let None = val {
                    *val = Some(
                        net.get_node(idx)
                            .params
                            .get_random_residence_time(
                                net.get_node(idx).params.state_to_index(&current_state[idx]),
                                net.get_param_index_network(idx, &current_state),
                                &mut rng,
                            )
                            .unwrap()
                            + t,
                    );
                }
            }

            let next_node_transition = next_transitions
                .iter()
                .enumerate()
                .min_by(|x, y| x.1.unwrap().partial_cmp(&y.1.unwrap()).unwrap())
                .unwrap()
                .0;
            if next_transitions[next_node_transition].unwrap() > t_end {
                break;
            }
            t = next_transitions[next_node_transition].unwrap().clone();
            time.push(t.clone());

            current_state[next_node_transition] = net
                .get_node(next_node_transition)
                .params
                .get_random_state(
                    net.get_node(next_node_transition)
                        .params
                        .state_to_index(&current_state[next_node_transition]),
                    net.get_param_index_network(next_node_transition, &current_state),
                    &mut rng,
                )
                .unwrap();

            events.push(Array::from_vec(
                current_state
                    .iter()
                    .map(|x| match x {
                        params::StateType::Discrete(state) => state.clone(),
                    })
                    .collect(),
            ));
            next_transitions[next_node_transition] = None;

            for child in net.get_children_set(next_node_transition) {
                next_transitions[child] = None
            }
        }

        events.push(
            current_state
                .iter()
                .map(|x| match x {
                    params::StateType::Discrete(state) => state.clone(),
                })
                .collect(),
        );
        time.push(t_end.clone());

        trajectories.push(Trajectory::init(
            Array::from_vec(time),
            Array2::from_shape_vec(
                (events.len(), current_state.len()),
                events.iter().flatten().cloned().collect(),
            )
            .unwrap(),
        ));
    }
    Dataset::init(trajectories)
}
