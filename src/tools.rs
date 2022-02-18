use ndarray::prelude::*;
use petgraph::prelude::*;
use crate::network;
use crate::node;
use crate::params::Params;

pub struct Trajectory {
    time: Array1<f64>,
    events: Array2<u32>
}

pub struct Dataset {
    trajectories: Vec<Trajectory>
}


fn trajectory_generator(net: &Box<dyn network::Network>, n_trajectories: u64, t_end: f64) -> Dataset {
    let mut dataset = Dataset{
        trajectories: Vec::new()
    };

    let node_idx: Vec<_> = net.get_node_indices().collect();
    for _ in 0..n_trajectories {
        let t = 0.0;
        let mut time: Vec<f64> = Vec::new();
        let mut events: Vec<Vec<u32>> = Vec::new();
        let mut current_state: Vec<u32> = node_idx.iter().map(|x| {
            match net.get_node_weight(&x).get_params() {
                node::NodeType::DiscreteStatesContinousTime(params) => 
                    params.get_random_state_uniform()
}
        }).collect();
        let next_transitions: Vec<Option<f64>> = (0..node_idx.len()).map(|_| Option::None).collect();
        events.push(current_state.clone());
        time.push(t.clone());
        while t < t_end {
            
        }



    }
    
    dataset
}
