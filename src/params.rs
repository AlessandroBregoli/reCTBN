use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use rand::Rng;
use std::collections::{BTreeSet, HashMap};
use thiserror::Error;

/// Error types for trait Params
#[derive(Error, Debug, PartialEq)]
pub enum ParamsError {
    #[error("Unsupported method")]
    UnsupportedMethod(String),
    #[error("Paramiters not initialized")]
    ParametersNotInitialized(String),
    #[error("Invalid cim for parameter")]
    InvalidCIM(String),
}

/// Allowed type of states
#[derive(Clone)]
pub enum StateType {
    Discrete(usize),
}

/// Parameters
/// The Params trait is the core element for building different types of nodes. The goal is to
/// define the set of method required to describes a generic node.
#[enum_dispatch(Params)]
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

    /// Validate parameters against domain
    fn validate_params(&self) -> Result<(), ParamsError>;
}

/// The Params enum is the core element for building different types of nodes. The goal is to
/// define all the supported type of parameters.
#[enum_dispatch]
pub enum Params {
    DiscreteStatesContinousTime(DiscreteStatesContinousTimeParams),
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
    pub domain: BTreeSet<String>,
    pub cim: Option<Array3<f64>>,
    pub transitions: Option<Array3<u64>>,
    pub residence_time: Option<Array2<f64>>,
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
        StateType::Discrete(rng.gen_range(0..(self.domain.len())))
    }

    fn get_random_residence_time(&self, state: usize, u: usize) -> Result<f64, ParamsError> {
        // Generate a random residence time given the current state of the node and its parent set.
        // The method used is described in:
        // https://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
        match &self.cim {
            Option::Some(cim) => {
                let mut rng = rand::thread_rng();
                let lambda = cim[[u, state, state]] * -1.0;
                let x: f64 = rng.gen_range(0.0..=1.0);
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
                let urand: f64 = rng.gen_range(0.0..=1.0);

                let next_state = cim.slice(s![u, state, ..]).map(|x| x / lambda).iter().fold(
                    (0, 0.0),
                    |mut acc, ele| {
                        if &acc.1 + ele < urand && ele > &0.0 {
                            acc.0 += 1;
                        }
                        if ele > &0.0 {
                            acc.1 += ele;
                        }
                        acc
                    },
                );

                let next_state = if next_state.0 < state {
                    next_state.0
                } else {
                    next_state.0 + 1
                };

                Ok(StateType::Discrete(next_state))
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

    fn validate_params(&self) -> Result<(), ParamsError> {
        let domain_size = self.domain.len();

        // Check if the cim is initialized
        if let None = self.cim {
            return Err(ParamsError::ParametersNotInitialized(String::from(
                "CIM not initialized",
            )));
        }
        let cim = self.cim.as_ref().unwrap();
        // Check if the inner dimensions of the cim are equal to the cardinality of the variable
        if cim.shape()[1] != domain_size || cim.shape()[2] != domain_size {
            return Err(ParamsError::InvalidCIM(format!(
                "Incompatible shape {:?} with domain {:?}",
                cim.shape(),
                domain_size
            )));
        }

        // Check if the diagonal of each cim is non-positive
        if cim
            .axis_iter(Axis(0))
            .any(|x| x.diag().iter().any(|x| x >= &0.0))
        {
            return Err(ParamsError::InvalidCIM(String::from(
                "The diagonal of each cim must be non-positive",
            )));
        }

        // Check if each row sum up to 0
        let zeros = Array::zeros(domain_size);
        if cim
            .axis_iter(Axis(0))
            .any(|x| !x.sum_axis(Axis(1)).abs_diff_eq(&zeros, f64::MIN_POSITIVE))
        {
            return Err(ParamsError::InvalidCIM(String::from(
                "The sum of each row must be 0",
            )));
        }

        return Ok(());
    }
}
