use ndarray::prelude::*;
use rand::Rng;
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;
use enum_dispatch::enum_dispatch;

/// Error types for trait Params
#[derive(Error, Debug)]
pub enum ParamsError {
    #[error("Unsupported method")]
    UnsupportedMethod(String),
    #[error("Paramiters not initialized")]
    ParametersNotInitialized(String),
}

/// Allowed type of states
#[derive(Clone)]
pub enum StateType {
    Discrete(u32),
}

/// Parameters
/// The Params trait is the core element for building different types of nodes. The goal is to
/// define the set of method required to describes a generic node.
pub trait ParamsTrait {
    fn reset_params(&mut self);

    /// Randomly generate a possible state of the node disregarding the state of the node and it's
    /// parents.
    fn get_random_state_uniform(&self) -> StateType;

    /// Randomly generate a residence time for the given node taking into account the node state
    /// and its parent set.
    fn get_random_residence_time(&self, state: usize, u: usize) -> Result<f64, ParamsError>;

    /// Randomly generate a possible state for the given node taking into account the node state
    /// and its parent set.
    fn get_random_state(&self, state: usize, u: usize) -> Result<StateType, ParamsError>;

    /// Used by childern of the node described by this parameters to reserve spaces in their CIMs.
    fn get_reserved_space_as_parent(&self) -> usize;

    /// Index used by discrete node to represents their states as usize.
    fn state_to_index(&self, state: &StateType) -> usize;
}

/// The Params enum is the core element for building different types of nodes. The goal is to
/// define all the supported type of parameters.
pub enum Params {
    DiscreteStatesContinousTime(DiscreteStatesContinousTimeParams),
}

impl ParamsTrait for Params {
    fn reset_params(&mut self) {
        match self {
            Params::DiscreteStatesContinousTime(p) => p.reset_params()
        }
    }

    fn get_random_state_uniform(&self) -> StateType{
        match self {
            Params::DiscreteStatesContinousTime(p) => p.get_random_state_uniform()
        }
    }


    /// Randomly generate a residence time for the given node taking into account the node state
    /// and its parent set.
    fn get_random_residence_time(&self, state: usize, u: usize) -> Result<f64, ParamsError> {
        match self {
            Params::DiscreteStatesContinousTime(p) => p.get_random_residence_time(state, u)
        }
    }


    /// Randomly generate a possible state for the given node taking into account the node state
    /// and its parent set.
    fn get_random_state(&self, state: usize, u: usize) -> Result<StateType, ParamsError> {
        match self {
            Params::DiscreteStatesContinousTime(p) => p.get_random_state(state, u)
        }
    }


    /// Used by childern of the node described by this parameters to reserve spaces in their CIMs.
    fn get_reserved_space_as_parent(&self) -> usize {
        match self {
            Params::DiscreteStatesContinousTime(p) => p.get_reserved_space_as_parent()
        }
    }


    /// Index used by discrete node to represents their states as usize.
    fn state_to_index(&self, state: &StateType) -> usize {
        match self {
            Params::DiscreteStatesContinousTime(p) => p.state_to_index(state)
        }
    }


}

/// DiscreteStatesContinousTime.
/// This represents the parameters of a classical discrete node for ctbn and it's composed by the
/// following elements:
/// - **domain**: an ordered and exhaustive set of possible states
/// - **cim**: Conditional Intensity Matrix
/// - **Sufficient Statistics**: the sufficient statistics are mainly used during the parameter
///     learning task and are composed by:
///     - **transitions**: number of transitions from one state to another given a specific
///     realization of the parent set
///     - **residence_time**: permanence time in each possible states given a specific
///     realization of the parent set
pub struct DiscreteStatesContinousTimeParams {
    domain: BTreeSet<String>,
    cim: Option<Array3<f64>>,
    transitions: Option<Array3<u64>>,
    residence_time: Option<Array2<f64>>,
}

impl DiscreteStatesContinousTimeParams {
    pub fn init(domain: BTreeSet<String>) -> DiscreteStatesContinousTimeParams {
        DiscreteStatesContinousTimeParams {
            domain: domain,
            cim: Option::None,
            transitions: Option::None,
            residence_time: Option::None,
        }
    }
}

impl ParamsTrait for DiscreteStatesContinousTimeParams {
    fn reset_params(&mut self) {
        self.cim = Option::None;
        self.transitions = Option::None;
        self.residence_time = Option::None;
    }

    fn get_random_state_uniform(&self) -> StateType {
        let mut rng = rand::thread_rng();
        StateType::Discrete(rng.gen_range(0..(self.domain.len() as u32)))
    }

    fn get_random_residence_time(&self, state: usize, u: usize) -> Result<f64, ParamsError> {
        // Generate a random residence time given the current state of the node and its parent set.
        // The method used is described in:
        // https://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
        match &self.cim {
            Option::Some(cim) => {
                let mut rng = rand::thread_rng();
                let lambda = cim[[u, state, state]] * -1.0;
                let x: f64 = rng.gen_range(0.0..1.0);
                Ok(-x.ln() / lambda)
            }
            Option::None => Err(ParamsError::ParametersNotInitialized(String::from(
                "CIM not initialized",
            ))),
        }
    }

    fn get_random_state(&self, state: usize, u: usize) -> Result<StateType, ParamsError> {
        // Generate a random transition given the current state of the node and its parent set.
        // The method used is described in:
        // https://en.wikipedia.org/wiki/Multinomial_distribution#Sampling_from_a_multinomial_distribution
        match &self.cim {
            Option::Some(cim) => {
                let mut rng = rand::thread_rng();
                let lambda = cim[[u, state, state]] * -1.0;
                let x: f64 = rng.gen_range(0.0..1.0);

                let next_state = cim.slice(s![u, state, ..]).map(|x| x / lambda).iter().fold(
                    (0, 0.0),
                    |mut acc, ele| {
                        if &acc.1 + ele < x && ele > &0.0 {
                            acc.1 += x;
                            acc.0 += 1;
                        }
                        acc
                    },
                );

                let next_state = if next_state.0 < state {
                    next_state.0
                } else {
                    next_state.0 + 1
                };

                Ok(StateType::Discrete(next_state as u32))
            }
            Option::None => Err(ParamsError::ParametersNotInitialized(String::from(
                "CIM not initialized",
            ))),
        }
    }

    fn get_reserved_space_as_parent(&self) -> usize {
        self.domain.len()
    }

    fn state_to_index(&self, state: &StateType) -> usize {
        match state {
            StateType::Discrete(val) => val.clone() as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use ndarray::prelude::*;

    fn create_ternary_discrete_time_continous_param() -> DiscreteStatesContinousTimeParams {
        let mut domain = BTreeSet::new();
        domain.insert(String::from("A"));
        domain.insert(String::from("B"));
        domain.insert(String::from("C"));
        let mut params = DiscreteStatesContinousTimeParams::init(domain);

        let cim = array![[[-3.0, 2.0, 1.0], [1.0, -5.0, 4.0], [3.2, 1.7, -4.0]]];

        params.cim = Some(cim);
        params
    }
    #[test]
    fn test_uniform_generation() {
        let param = create_ternary_discrete_time_continous_param();
        let mut states = Array1::<u32>::zeros(10000);

        states.mapv_inplace(|_| {
            if let StateType::Discrete(val) = param.get_random_state_uniform() {
                val
            } else {
                panic!()
            }
        });
        let zero_freq = states.mapv(|a| (a == 0) as u64).sum() as f64 / 10000.0;

        assert_relative_eq!(1.0 / 3.0, zero_freq, epsilon = 0.01);
    }

    #[test]
    fn test_random_generation_state() {
        let param = create_ternary_discrete_time_continous_param();
        let mut states = Array1::<u32>::zeros(10000);

        states.mapv_inplace(|_| {
            if let StateType::Discrete(val) = param.get_random_state(1, 0).unwrap() {
                val
            } else {
                panic!()
            }
        });
        let two_freq = states.mapv(|a| (a == 2) as u64).sum() as f64 / 10000.0;
        let zero_freq = states.mapv(|a| (a == 0) as u64).sum() as f64 / 10000.0;

        assert_relative_eq!(4.0 / 5.0, two_freq, epsilon = 0.01);
        assert_relative_eq!(1.0 / 5.0, zero_freq, epsilon = 0.01);
    }

    #[test]
    fn test_random_generation_residence_time() {
        let param = create_ternary_discrete_time_continous_param();
        let mut states = Array1::<f64>::zeros(10000);

        states.mapv_inplace(|_| param.get_random_residence_time(1, 0).unwrap());

        assert_relative_eq!(1.0 / 5.0, states.mean().unwrap(), epsilon = 0.01);
    }
}
