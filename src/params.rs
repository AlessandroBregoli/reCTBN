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
            let x: f64 = rng.gen_range(0.0..1.0);

            let next_state = cim.slice(s![u,state,..]).map(|x| x / lambda).iter().fold((0, 0.0), |mut acc, ele| { 
                if &acc.1 + ele < x  && ele > &0.0{
                    acc.1 += x;
                    acc.0 += 1;
                }                
                acc});

            let next_state = if next_state.0 < state {
                next_state.0
            } else {
                next_state.0 + 1
            };

            Ok(StateType::Discrete(next_state as u32))
            
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;


    fn create_ternary_discrete_time_continous_param() -> DiscreteStatesContinousTimeParams {
        let mut domain = BTreeSet::new();
        domain.insert(String::from("A"));
        domain.insert(String::from("B"));
        domain.insert(String::from("C"));
        let mut params = DiscreteStatesContinousTimeParams::init(domain);

        let cim = array![
            [
                [-3.0, 2.0, 1.0],
                [1.0, -5.0, 4.0],
                [3.2, 1.7, -4.0]
            ]];

        params.cim = Some(cim);
        params
    }
    
    #[test]
    fn test_uniform_generation() {
        let param = create_ternary_discrete_time_continous_param();
        let mut states = Array1::<u32>::zeros(10000);

        states.mapv_inplace(|_| if let StateType::Discrete(val) = param.get_random_state_uniform() {
            val
        } else {panic!()});
        let zero_freq = states.mapv(|a| (a == 0) as u64).sum() as f64 / 10000.0;

        assert_relative_eq!(1.0/3.0, zero_freq, epsilon=0.01);
    }


    #[test]
    fn test_random_generation_state() {
        let param = create_ternary_discrete_time_continous_param();
        let mut states = Array1::<u32>::zeros(10000);

        states.mapv_inplace(|_| if let StateType::Discrete(val) = param.get_random_state(1, 0).unwrap() {
            val
        } else {panic!()});
        let two_freq = states.mapv(|a| (a == 2) as u64).sum() as f64 / 10000.0;
        let zero_freq = states.mapv(|a| (a == 0) as u64).sum() as f64 / 10000.0;

        assert_relative_eq!(4.0/5.0, two_freq, epsilon=0.01);
        assert_relative_eq!(1.0/5.0, zero_freq, epsilon=0.01);
    }

    
    #[test]
    fn test_random_generation_residence_time() {
        let param = create_ternary_discrete_time_continous_param();
        let mut states = Array1::<f64>::zeros(10000);

        states.mapv_inplace(|_| param.get_random_residence_time(1, 0).unwrap() );

        assert_relative_eq!(1.0/5.0, states.mean().unwrap(), epsilon=0.01);

    }

}
