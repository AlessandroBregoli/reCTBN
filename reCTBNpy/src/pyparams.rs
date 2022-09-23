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
pub struct PyDiscreteStateContinousTime {
    param: params::DiscreteStatesContinousTimeParams,
}

#[pymethods]
impl PyDiscreteStateContinousTime {
    #[new]
    pub fn new(label: String, domain: BTreeSet<String>) -> Self {
        PyDiscreteStateContinousTime {
            param: params::DiscreteStatesContinousTimeParams::new(label, domain),
        }
    }

    pub fn get_cim<'py>(&self, py: Python<'py>) -> Option<&'py numpy::PyArray3<f64>> {
        match self.param.get_cim() {
            Some(x) => Some(x.to_pyarray(py)),
            None => None,
        }
    }

    pub fn set_cim<'py>(&mut self, py: Python<'py>, cim: numpy::PyReadonlyArray3<f64>) -> Result<(), PyParamsError> {
        self.param.set_cim(cim.as_array().to_owned())?;
        Ok(())
    }

    pub fn get_label(&self) -> String {
        self.param.get_label().to_string()
    }
}
