use pyo3::prelude::*;
pub mod pyctbn;
pub mod pyparams;



/// A Python module implemented in Rust.
#[pymodule]
fn reCTBNpy(py: Python, m: &PyModule) -> PyResult<()> {
    let network_module = PyModule::new(py, "network")?;
    network_module.add_class::<pyctbn::PyCtbnNetwork>()?;
    m.add_submodule(network_module)?;

    let params_module = PyModule::new(py, "params")?;
    params_module.add_class::<pyparams::PyDiscreteStateContinousTime>()?;
    m.add_submodule(params_module)?;
    Ok(())
}
