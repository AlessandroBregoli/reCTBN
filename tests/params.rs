use rustyCTBN::params::*;
use ndarray::prelude::*;
use std::collections::BTreeSet;

mod utils;

#[macro_use]
extern crate approx;


fn create_ternary_discrete_time_continous_param() -> DiscreteStatesContinousTimeParams {
    let mut params = utils::generate_discrete_time_continous_param(3);

    let cim = array![[[-3.0, 2.0, 1.0], [1.0, -5.0, 4.0], [3.2, 1.7, -4.0]]];

    params.cim = Some(cim);
    params
}

#[test]
fn test_uniform_generation() {
    let param = create_ternary_discrete_time_continous_param();
    let mut states = Array1::<usize>::zeros(10000);

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
    let mut states = Array1::<usize>::zeros(10000);

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
