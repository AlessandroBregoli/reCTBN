use crate::network;
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
    pub fn new(time: Array1<f64>, events: Array2<usize>) -> Trajectory {
        //Events and time are two part of the same trajectory. For this reason they must have the
        //same number of sample.
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
    pub fn new(trajectories: Vec<Trajectory>) -> Dataset {

        //All the trajectories in the same dataset must represent the same process. For this reason
        //each trajectory must represent the same number of variables.
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
    
    //Tmp growing vector containing generated trajectories.
    let mut trajectories: Vec<Trajectory> = Vec::new();
    
    //Random Generator object
    let mut rng: ChaCha8Rng = match seed {
        //If a seed is present use it to initialize the random generator.
        Some(seed) => SeedableRng::seed_from_u64(seed),
        //Otherwise create a new random generator using the method `from_entropy`
        None => SeedableRng::from_entropy()
    };
    
    //Each iteration generate one trajectory
    for _ in 0..n_trajectories {
        //Current time of the sampling process
        let mut t = 0.0;
        //History of all the moments in which something changed
        let mut time: Vec<f64> = Vec::new();
        //Configuration of the process variables at time t initialized with an uniform
        //distribution.
        let mut current_state: Vec<params::StateType> = net.get_node_indices()
            .map(|x| net.get_node(x).get_random_state_uniform(&mut rng))
            .collect();
        //History of all the configurations of the process variables. 
        let mut events: Vec<Array1<usize>> = Vec::new();
        //Vector containing to time to the next transition for each variable.
        let mut next_transitions: Vec<Option<f64>> =
            net.get_node_indices().map(|_| Option::None).collect();
        
        //Add the starting time for the trajectory.
        time.push(t.clone());
        //Add the starting configuration of the trajectory.
        events.push(
            current_state
                .iter()
                .map(|x| match x {
                    params::StateType::Discrete(state) => state.clone(),
                })
                .collect(),
        );
        //Generate new samples until ending time is reached.
        while t < t_end {
            //Generate the next transition time for each uninitialized variable.
            for (idx, val) in next_transitions.iter_mut().enumerate() {
                if let None = val {
                    *val = Some(
                        net.get_node(idx)
                            .get_random_residence_time(
                                net.get_node(idx).state_to_index(&current_state[idx]),
                                net.get_param_index_network(idx, &current_state),
                                &mut rng,
                            )
                            .unwrap()
                            + t,
                    );
                }
            }
            
            //Get the variable with the smallest transition time.
            let next_node_transition = next_transitions
                .iter()
                .enumerate()
                .min_by(|x, y| x.1.unwrap().partial_cmp(&y.1.unwrap()).unwrap())
                .unwrap()
                .0;
            //Check if the next transition take place after the ending time.
            if next_transitions[next_node_transition].unwrap() > t_end {
                break;
            }
            //Get the time in which the next transition occurs.
            t = next_transitions[next_node_transition].unwrap().clone();
            //Add the transition time to next
            time.push(t.clone());
            
            //Compute the new state of the transitioning variable.
            current_state[next_node_transition] = net
                .get_node(next_node_transition)
                .get_random_state(
                    net.get_node(next_node_transition)
                        .state_to_index(&current_state[next_node_transition]),
                    net.get_param_index_network(next_node_transition, &current_state),
                    &mut rng,
                )
                .unwrap();
            
            //Add the new state to events
            events.push(Array::from_vec(
                current_state
                    .iter()
                    .map(|x| match x {
                        params::StateType::Discrete(state) => state.clone(),
                    })
                    .collect(),
            ));
            //Reset the next transition time for the transitioning node.
            next_transitions[next_node_transition] = None;

            //Reset the next transition time for each child of the transitioning node.
            for child in net.get_children_set(next_node_transition) {
                next_transitions[child] = None
            }
        }
        
        //Add current_state as last state.
        events.push(
            current_state
                .iter()
                .map(|x| match x {
                    params::StateType::Discrete(state) => state.clone(),
                })
                .collect(),
        );
        //Add t_end as last time.
        time.push(t_end.clone());
        
        //Add the sampled trajectory to trajectories.
        trajectories.push(Trajectory::new(
            Array::from_vec(time),
            Array2::from_shape_vec(
                (events.len(), current_state.len()),
                events.iter().flatten().cloned().collect(),
            )
            .unwrap(),
        ));
    }
    //Return a dataset object with the sampled trajectories.
    Dataset::new(trajectories)
}
