use std::collections::BTreeSet;

use reCTBN::params;

#[allow(dead_code)]
pub fn generate_discrete_time_continous_node(label: String, cardinality: usize) -> params::Params {
    params::Params::DiscreteStatesContinousTime(generate_discrete_time_continous_params(
        label,
        cardinality,
    ))
}

pub fn generate_discrete_time_continous_params(
    label: String,
    cardinality: usize,
) -> params::DiscreteStatesContinousTimeParams {
    let domain: BTreeSet<String> = (0..cardinality).map(|x| x.to_string()).collect();
    params::DiscreteStatesContinousTimeParams::new(label, domain)
}
