use ndarray::prelude::*;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
use reCTBN::params::{ParamsTrait, *};

mod utils;

#[macro_use]
extern crate approx;


fn create_ternary_discrete_time_continous_param() -> DiscreteStatesContinousTimeParams {
    let mut params = utils::generate_discrete_time_continous_params("A".to_string(), 3);

    let cim = array![[[-3.0, 2.0, 1.0], [1.0, -5.0, 4.0], [2.3, 1.7, -4.0]]];

    params.set_cim(cim);
    params
}

#[test]
fn test_get_label() {
    let param = create_ternary_discrete_time_continous_param();
    assert_eq!(&String::from("A"), param.get_label())
}

#[test]
fn test_uniform_generation() {
    let param = create_ternary_discrete_time_continous_param();
    let mut states = Array1::<usize>::zeros(10000);

    let mut rng = ChaCha8Rng::seed_from_u64(6347747169756259);

    states.mapv_inplace(|_| {
        if let StateType::Discrete(val) = param.get_random_state_uniform(&mut rng) {
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

    let mut rng = ChaCha8Rng::seed_from_u64(6347747169756259);

    states.mapv_inplace(|_| {
        if let StateType::Discrete(val) = param.get_random_state(1, 0, &mut rng).unwrap() {
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

    let mut rng = ChaCha8Rng::seed_from_u64(6347747169756259);

    states.mapv_inplace(|_| param.get_random_residence_time(1, 0, &mut rng).unwrap());

    assert_relative_eq!(1.0 / 5.0, states.mean().unwrap(), epsilon = 0.01);
}

#[test]
fn test_validate_params_valid_cim() {
    let param = create_ternary_discrete_time_continous_param();

    assert_eq!(Ok(()), param.validate_params());
}

#[test]
fn test_validate_params_valid_cim_with_huge_values() {
    let mut param = utils::generate_discrete_time_continous_params("A".to_string(), 3);
    let cim = array![[
        [-2e10, 1e10, 1e10],
        [1.5e10, -3e10, 1.5e10],
        [1e10, 1e10, -2e10]
    ]];
    let result = param.set_cim(cim);
    assert_eq!(Ok(()), result);
}

#[test]
fn test_validate_params_cim_not_initialized() {
    let param = utils::generate_discrete_time_continous_params("A".to_string(), 3);
    assert_eq!(
        Err(ParamsError::ParametersNotInitialized(String::from(
            "CIM not initialized",
        ))),
        param.validate_params()
    );
}

#[test]
fn test_validate_params_wrong_shape() {
    let mut param = utils::generate_discrete_time_continous_params("A".to_string(), 4);
    let cim = array![[[-3.0, 2.0, 1.0], [1.0, -5.0, 4.0], [2.3, 1.7, -4.0]]];
    let result = param.set_cim(cim);
    assert_eq!(
        Err(ParamsError::InvalidCIM(String::from(
            "Incompatible shape [1, 3, 3] with domain 4"
        ))),
        result
    );
}

#[test]
fn test_validate_params_positive_diag() {
    let mut param = utils::generate_discrete_time_continous_params("A".to_string(), 3);
    let cim = array![[[2.0, -3.0, 1.0], [1.0, -5.0, 4.0], [2.3, 1.7, -4.0]]];
    let result = param.set_cim(cim);
    assert_eq!(
        Err(ParamsError::InvalidCIM(String::from(
            "The diagonal of each cim must be non-positive",
        ))),
        result
    );
}

#[test]
fn test_validate_params_row_not_sum_to_zero() {
    let mut param = utils::generate_discrete_time_continous_params("A".to_string(), 3);
    let cim = array![[[-3.0, 2.0, 1.0], [1.0, -5.0, 4.0], [2.3, 1.701, -4.0]]];
    let result = param.set_cim(cim);
    assert_eq!(
        Err(ParamsError::InvalidCIM(String::from(
            "The sum of each row must be 0"
        ))),
        result
    );
}
