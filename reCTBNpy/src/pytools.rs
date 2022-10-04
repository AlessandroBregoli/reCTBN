use numpy::{self, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*};
use reCTBN::{tools, network};

#[pyclass]
#[derive(Clone)]
pub struct PyTrajectory(pub tools::Trajectory);

#[pymethods]
impl PyTrajectory {
    #[new]
    pub fn new(
        time: numpy::PyReadonlyArray1<f64>,
        events: numpy::PyReadonlyArray2<usize>,
    ) -> PyTrajectory {
        PyTrajectory(tools::Trajectory::new(
            time.as_array().to_owned(),
            events.as_array().to_owned(),
        ))
    }

    pub fn get_time<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray1<f64> {
        self.0.get_time().to_pyarray(py)
    }
    
    pub fn get_events<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray2<usize> {
        self.0.get_events().to_pyarray(py)
    }
}

#[pyclass]
pub struct PyDataset(pub tools::Dataset);

#[pymethods]
impl PyDataset {
    #[new]
    pub fn new(trajectories: Vec<PyTrajectory>) -> PyDataset {
        PyDataset(tools::Dataset::new(trajectories.into_iter().map(|x| x.0).collect()))
    }

    pub fn get_number_of_trajectories(&self) -> usize {
        self.0.get_trajectories().len()
    }

    pub fn get_trajectory(&self, idx: usize) -> PyTrajectory {
        PyTrajectory(self.0.get_trajectories().get(idx).unwrap().clone())
    }

}

