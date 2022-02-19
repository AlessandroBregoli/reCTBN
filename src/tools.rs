use ndarray::prelude::*;
use crate::network;
use crate::node;
use crate::params;
use crate::params::Params;

pub struct Trajectory {
    time: Array1<f64>,
    events: Array2<u32>
}

pub struct Dataset {
    trajectories: Vec<Trajectory>
}


pub fn trajectory_generator(net: &Box<dyn network::Network>, n_trajectories: u64, t_end: f64) -> Dataset {
    let mut dataset = Dataset{
        trajectories: Vec::new()
    };

    let node_idx: Vec<_> = net.get_node_indices().collect();
    for _ in 0..n_trajectories {
        let mut t = 0.0;
        let mut time: Vec<f64> = Vec::new();
        let mut events: Vec<Vec<params::StateType>> = Vec::new();
        let mut current_state: Vec<params::StateType> = node_idx.iter().map(|x| {
            net.get_node(*x).get_random_state_uniform()
        }).collect();
        let mut next_transitions: Vec<Option<f64>> = (0..node_idx.len()).map(|_| Option::None).collect();
        events.push(current_state.clone());
        time.push(t.clone());
        while t < t_end {
            next_transitions.iter_mut().enumerate().map(|(idx, val)| {
                if let None = val {
                    *val = Some(net.get_node(idx)
                                .get_random_residence_time(net.get_node(idx).state_to_index(&current_state[idx]), 
                                                           net.get_param_index_network(idx, &current_state)).unwrap() + t);
                }
            });
            let next_node_transition = next_transitions
                .iter()
                .enumerate()
                .min_by(|x, y| 
                        x.1.unwrap().partial_cmp(&y.1.unwrap()).unwrap())
                .unwrap().0;
            
            if next_transitions[next_node_transition].unwrap() > t_end {
                break
            }
            
            t = next_transitions[next_node_transition].unwrap().clone();
            time.push(t.clone());

            current_state[next_node_transition] = net.get_node(next_node_transition)
                                .get_random_state(
                                    net.get_node(next_node_transition).
                                    state_to_index(
                                        &current_state[next_node_transition]),
                                        net.get_param_index_network(next_node_transition, &current_state))
                                .unwrap();

            events.push(current_state.clone());
            next_transitions[next_node_transition] = None;
            
            for child in net.get_children_set(next_node_transition){
                next_transitions[child] = None
            }



        }



    }
    
    dataset
}
