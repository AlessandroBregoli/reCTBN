use ndarray::prelude::*;

pub struct CIM {
    cim: Array3<f64>,
}

pub struct M {
    transitions: Array3<u64>,
}

pub struct T {
    residence_time: Array2<f64>,
}


