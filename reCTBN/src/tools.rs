//! Contains commonly used methods used across the crate.

use ndarray::prelude::*;

use crate::sampling::{ForwardSampler, Sampler};
use crate::{params, process};

#[derive(Clone)]
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

#[derive(Clone)]
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

pub fn trajectory_generator<T: process::NetworkProcess>(
    net: &T,
    n_trajectories: u64,
    t_end: f64,
    seed: Option<u64>,
) -> Dataset {
    //Tmp growing vector containing generated trajectories.
    let mut trajectories: Vec<Trajectory> = Vec::new();

    //Random Generator object
    let mut sampler = ForwardSampler::new(net, seed, None);
    //Each iteration generate one trajectory
    for _ in 0..n_trajectories {
        //History of all the moments in which something changed
        let mut time: Vec<f64> = Vec::new();
        //Configuration of the process variables at time t initialized with an uniform
        //distribution.
        let mut events: Vec<process::NetworkProcessState> = Vec::new();

        //Current Time and Current State
        let mut sample = sampler.next().unwrap();
        //Generate new samples until ending time is reached.
        while sample.t < t_end {
            time.push(sample.t);
            events.push(sample.state);
            sample = sampler.next().unwrap();
        }

        let current_state = events.last().unwrap().clone();
        events.push(current_state);

        //Add t_end as last time.
        time.push(t_end.clone());

        //Add the sampled trajectory to trajectories.
        trajectories.push(Trajectory::new(
            Array::from_vec(time),
            Array2::from_shape_vec(
                (events.len(), events.last().unwrap().len()),
                events
                    .iter()
                    .flatten()
                    .map(|x| match x {
                        params::StateType::Discrete(x) => x.clone(),
                    })
                    .collect(),
            )
            .unwrap(),
        ));
        sampler.reset();
    }
    //Return a dataset object with the sampled trajectories.
    Dataset::new(trajectories)
}
