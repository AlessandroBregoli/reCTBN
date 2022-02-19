use ndarray::prelude::*;
use std::collections::{HashMap, BTreeSet};
use rand::Rng;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParamsError {
    #[error("Unsupported method")]
    UnsupportedMethod(String),
    #[error("Paramiters not initialized")]
    ParametersNotInitialized(String)
}

pub trait Params {
    fn reset_params(&mut self);
    fn get_random_state_uniform(&self) -> StateType;
    fn get_random_residence_time(&self, state: usize, u: usize) -> Result<f64, ParamsError>;
    fn get_random_state(&self, state: usize, u:usize) -> Result<StateType, ParamsError>;
    fn get_reserved_space_as_parent(&self) -> usize;
    fn state_to_index(&self, state: &StateType) -> usize;
}

#[derive(Clone)]
pub enum StateType {
    Discrete(u32)
}

pub struct DiscreteStatesContinousTimeParams {
    domain: BTreeSet<String>,
    cim: Option<Array3<f64>>,
    transitions: Option<Array3<u64>>,
    residence_time: Option<Array2<f64>>
}

impl DiscreteStatesContinousTimeParams  {
    pub fn init(domain: BTreeSet<String>) -> DiscreteStatesContinousTimeParams {
        DiscreteStatesContinousTimeParams {
            domain: domain,
            cim: Option::None,
            transitions: Option::None,
            residence_time: Option::None
        }
    }
}
impl Params for DiscreteStatesContinousTimeParams {

    fn reset_params(&mut self) {
        self.cim = Option::None;
        self.transitions = Option::None;
        self.residence_time = Option::None;
    }

    fn get_random_state_uniform(&self) -> StateType {
        let mut rng = rand::thread_rng();
        StateType::Discrete(rng.gen_range(0..(self.domain.len() as u32)))
    }

    fn get_random_residence_time(&self, state: usize, u:usize) -> Result<f64, ParamsError> {
       match &self.cim {
           Option::Some(cim) => {   
            let mut rng = rand::thread_rng();
            let lambda = cim[[u, state, state]] * -1.0;
            let x:f64 = rng.gen_range(0.0..1.0);
            Ok(-x.ln()/lambda)
           },
           Option::None => Err(ParamsError::ParametersNotInitialized(String::from("CIM not initialized")))
       }
    }


    fn get_random_state(&self, state: usize, u:usize) -> Result<StateType, ParamsError>{
       match &self.cim {
           Option::Some(cim) => {   
            let mut rng = rand::thread_rng();
            let lambda = cim[[u, state, state]] * -1.0;
            let x = rng.gen_range(0.0..1.0);

            let state = (cim.slice(s![u,state,..])).iter().scan((0, 0.0), |acc, &x| {
                if x > 0.0 && acc.1 < x {
                    acc.0 += 1;
                    acc.1 += x;
                    return Some(*acc);
                } else if acc.1 < x {
                    acc.0 += 1;
                    return Some(*acc);
                }
                None

            }).last();
            Ok(StateType::Discrete(state.unwrap().0))
            
           },
           Option::None => Err(ParamsError::ParametersNotInitialized(String::from("CIM not initialized")))
       }
    }


    fn get_reserved_space_as_parent(&self) -> usize {
        self.domain.len()
    }

    fn state_to_index(&self, state: &StateType) -> usize {
        match state {
            StateType::Discrete(val) => val.clone() as usize
        }
    }
}
