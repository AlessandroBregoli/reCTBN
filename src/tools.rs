use ndarray::prelude::*;
use crate::network;
use petgraph::prelude::*;
use rand::Rng;

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
    for _ in 0..n_trajectories {
        let mut rng = rand::thread_rng();
        let t = 0.0;
        let mut time: Vec<f64> = Vec::new();
        let mut events: Vec<Vec<u32>> = Vec::new();
        let current_state: Vec<u32> = net.get_node_indices().map(|x| rng.gen_range(0..2)).collect();

    }
    
    dataset
}
