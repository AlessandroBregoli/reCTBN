mod utils;

use approx::{abs_diff_eq, assert_abs_diff_eq};
use ndarray::*;
use reCTBN::{
    params,
    process::{ctbn::*, NetworkProcess, NetworkProcessState},
    reward::{reward_evaluation::*, reward_function::*, *},
};
use utils::generate_discrete_time_continous_node;

#[test]
fn simple_factored_reward_function_binary_node_MC() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
    rf.get_transition_reward_mut(n1)
        .assign(&arr2(&[[0.0, 0.0], [0.0, 0.0]]));
    rf.get_instantaneous_reward_mut(n1)
        .assign(&arr1(&[3.0, 3.0]));

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.set_cim(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]])).unwrap();
        }
    }

    net.initialize_adj_matrix();

    let s0: NetworkProcessState = vec![params::StateType::Discrete(0)];
    let s1: NetworkProcessState = vec![params::StateType::Discrete(1)];

    let mc = MonteCarloReward::new(10000, 1e-1, 1e-1, 10.0, RewardCriteria::InfiniteHorizon { discount_factor: 1.0 }, Some(215));
    assert_abs_diff_eq!(3.0, mc.evaluate_state(&net, &rf, &s0), epsilon = 1e-2);
    assert_abs_diff_eq!(3.0, mc.evaluate_state(&net, &rf, &s1), epsilon = 1e-2);
    
    let rst = mc.evaluate_state_space(&net, &rf);
    assert_abs_diff_eq!(3.0, rst[&s0], epsilon = 1e-2);
    assert_abs_diff_eq!(3.0, rst[&s1], epsilon = 1e-2);


    let mc = MonteCarloReward::new(10000, 1e-1, 1e-1, 10.0, RewardCriteria::FiniteHorizon, Some(215));
    assert_abs_diff_eq!(30.0, mc.evaluate_state(&net, &rf, &s0), epsilon = 1e-2);
    assert_abs_diff_eq!(30.0, mc.evaluate_state(&net, &rf, &s1), epsilon = 1e-2);
    

}

#[test]
fn simple_factored_reward_function_chain_MC() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();

    let n3 = net
        .add_node(generate_discrete_time_continous_node(String::from("n3"), 2))
        .unwrap();

    net.add_edge(n1, n2);
    net.add_edge(n2, n3);

    match &mut net.get_node_mut(n1) {
        params::Params::DiscreteStatesContinousTime(param) => {
            param.set_cim(arr3(&[[[-0.1, 0.1], [1.0, -1.0]]])).unwrap();
        }
    }

    match &mut net.get_node_mut(n2) {
        params::Params::DiscreteStatesContinousTime(param) => {
            param
                .set_cim(arr3(&[
                    [[-0.01, 0.01], [5.0, -5.0]],
                    [[-5.0, 5.0], [0.01, -0.01]],
                ]))
                .unwrap();
        }
    }


    match &mut net.get_node_mut(n3) {
        params::Params::DiscreteStatesContinousTime(param) => {
            param
                .set_cim(arr3(&[
                    [[-0.01, 0.01], [5.0, -5.0]],
                    [[-5.0, 5.0], [0.01, -0.01]],
                ]))
                .unwrap();
        }
    }


    let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
    rf.get_transition_reward_mut(n1)
        .assign(&arr2(&[[0.0, 1.0], [1.0, 0.0]]));

    rf.get_transition_reward_mut(n2)
        .assign(&arr2(&[[0.0, 1.0], [1.0, 0.0]]));

    rf.get_transition_reward_mut(n3)
        .assign(&arr2(&[[0.0, 1.0], [1.0, 0.0]]));

    let s000: NetworkProcessState = vec![
        params::StateType::Discrete(1),
        params::StateType::Discrete(0),
        params::StateType::Discrete(0),
    ];

    let mc = MonteCarloReward::new(10000, 1e-1, 1e-1, 10.0, RewardCriteria::InfiniteHorizon { discount_factor: 1.0 }, Some(215));
    assert_abs_diff_eq!(2.447, mc.evaluate_state(&net, &rf, &s000), epsilon = 1e-1);

    let rst = mc.evaluate_state_space(&net, &rf);
    assert_abs_diff_eq!(2.447, rst[&s000], epsilon = 1e-1);

}
