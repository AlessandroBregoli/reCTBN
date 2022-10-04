use numpy::{self, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*};
use reCTBN::params::{self, ParamsTrait};
use std::collections::BTreeSet;

pub struct PyParamsError(params::ParamsError);

impl From<PyParamsError> for PyErr {
    fn from(error: PyParamsError) -> Self {
        PyValueError::new_err(error.0.to_string())
    }
}

impl From<params::ParamsError> for PyParamsError {
    fn from(other: params::ParamsError) -> Self {
        Self(other)
    }
}

#[pyclass]
pub struct PyStateType(pub params::StateType);

#[pyclass]
#[derive(Clone)]
pub struct PyParams(pub params::Params);

#[pymethods]
impl PyParams {
    #[staticmethod]
    pub fn new_discrete_state_continous_time(p: PyDiscreteStateContinousTime) -> Self{
        PyParams(params::Params::DiscreteStatesContinousTime(p.0))
    }

    pub fn get_reserved_space_as_parent(&self) -> usize {
        self.0.get_reserved_space_as_parent()
    }

    pub fn get_label(&self) -> String {
        self.0.get_label().to_string()
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
#[derive(Clone)]
#[pyclass]
pub struct PyDiscreteStateContinousTime(params::DiscreteStatesContinousTimeParams);


#[pymethods]
impl PyDiscreteStateContinousTime {
    #[new]
    pub fn new(label: String, domain: BTreeSet<String>) -> Self {
        PyDiscreteStateContinousTime(params::DiscreteStatesContinousTimeParams::new(label, domain))
    }

    pub fn get_cim<'py>(&self, py: Python<'py>) -> Option<&'py numpy::PyArray3<f64>> {
        match self.0.get_cim() {
            Some(x) => Some(x.to_pyarray(py)),
            None => None,
        }
    }

    pub fn set_cim<'py>(&mut self, py: Python<'py>, cim: numpy::PyReadonlyArray3<f64>) -> Result<(), PyParamsError> {
        self.0.set_cim(cim.as_array().to_owned())?;
        Ok(())
    }


    pub fn get_transitions<'py>(&self, py: Python<'py>) -> Option<&'py numpy::PyArray3<usize>> {
        match self.0.get_transitions() {
            Some(x) => Some(x.to_pyarray(py)),
            None => None,
        }
    }

    pub fn set_transitions<'py>(&mut self, py: Python<'py>, cim: numpy::PyReadonlyArray3<usize>){
        self.0.set_transitions(cim.as_array().to_owned());
    }


    pub fn get_residence_time<'py>(&self, py: Python<'py>) -> Option<&'py numpy::PyArray2<f64>> {
        match self.0.get_residence_time() {
            Some(x) => Some(x.to_pyarray(py)),
            None => None,
        }
    }

    pub fn set_residence_time<'py>(&mut self, py: Python<'py>, cim: numpy::PyReadonlyArray2<f64>) {
        self.0.set_residence_time(cim.as_array().to_owned());
    }

}
