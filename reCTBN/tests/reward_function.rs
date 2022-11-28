mod utils;

use ndarray::*;
use utils::generate_discrete_time_continous_node;
use reCTBN::{process::{NetworkProcess, ctbn::*, NetworkProcessState}, reward::{*, reward_function::*}, params};


#[test]
fn simple_factored_reward_function_binary_node() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
    rf.get_transition_reward_mut(n1).assign(&arr2(&[[12.0, 1.0],[2.0,12.0]]));
    rf.get_instantaneous_reward_mut(n1).assign(&arr1(&[3.0,5.0]));
    
    let s0: NetworkProcessState = vec![params::StateType::Discrete(0)];
    let s1: NetworkProcessState =  vec![params::StateType::Discrete(1)];
    assert_eq!(rf.call(s0.clone(), None), Reward{transition_reward: 0.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s1.clone(), None), Reward{transition_reward: 0.0, instantaneous_reward: 5.0});


    assert_eq!(rf.call(s0.clone(), Some(s1.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s1.clone(), Some(s0.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 5.0});

    assert_eq!(rf.call(s0.clone(), Some(s0.clone())), Reward{transition_reward: 0.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s1.clone(), Some(s1.clone())), Reward{transition_reward: 0.0, instantaneous_reward: 5.0});
}


#[test]
fn simple_factored_reward_function_ternary_node() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 3))
        .unwrap();

    let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
    rf.get_transition_reward_mut(n1).assign(&arr2(&[[0.0, 1.0, 3.0],[2.0,0.0, 4.0], [5.0, 6.0, 0.0]]));
    rf.get_instantaneous_reward_mut(n1).assign(&arr1(&[3.0,5.0, 9.0]));
    
    let s0: NetworkProcessState = vec![params::StateType::Discrete(0)];
    let s1: NetworkProcessState = vec![params::StateType::Discrete(1)];
    let s2: NetworkProcessState = vec![params::StateType::Discrete(2)];


    assert_eq!(rf.call(s0.clone(), Some(s1.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s0.clone(), Some(s2.clone())), Reward{transition_reward: 5.0, instantaneous_reward: 3.0});


    assert_eq!(rf.call(s1.clone(), Some(s0.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 5.0});
    assert_eq!(rf.call(s1.clone(), Some(s2.clone())), Reward{transition_reward: 6.0, instantaneous_reward: 5.0});


    assert_eq!(rf.call(s2.clone(), Some(s0.clone())), Reward{transition_reward: 3.0, instantaneous_reward: 9.0});
    assert_eq!(rf.call(s2.clone(), Some(s1.clone())), Reward{transition_reward: 4.0, instantaneous_reward: 9.0});
}

#[test]
fn factored_reward_function_two_nodes() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 3))
        .unwrap();
    let n2 = net
        .add_node(generate_discrete_time_continous_node(String::from("n2"), 2))
        .unwrap();
    net.add_edge(n1, n2);


    let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
    rf.get_transition_reward_mut(n1).assign(&arr2(&[[0.0, 1.0, 3.0],[2.0,0.0, 4.0], [5.0, 6.0, 0.0]]));
    rf.get_instantaneous_reward_mut(n1).assign(&arr1(&[3.0,5.0, 9.0]));


    rf.get_transition_reward_mut(n2).assign(&arr2(&[[12.0, 1.0],[2.0,12.0]]));
    rf.get_instantaneous_reward_mut(n2).assign(&arr1(&[3.0,5.0]));
    
    let s00: NetworkProcessState = vec![params::StateType::Discrete(0), params::StateType::Discrete(0)];
    let s01: NetworkProcessState = vec![params::StateType::Discrete(1), params::StateType::Discrete(0)];
    let s02: NetworkProcessState = vec![params::StateType::Discrete(2), params::StateType::Discrete(0)];


    let s10: NetworkProcessState = vec![params::StateType::Discrete(0), params::StateType::Discrete(1)];
    let s11: NetworkProcessState = vec![params::StateType::Discrete(1), params::StateType::Discrete(1)];
    let s12: NetworkProcessState = vec![params::StateType::Discrete(2), params::StateType::Discrete(1)];

    assert_eq!(rf.call(s00.clone(), Some(s01.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 6.0});
    assert_eq!(rf.call(s00.clone(), Some(s02.clone())), Reward{transition_reward: 5.0, instantaneous_reward: 6.0});
    assert_eq!(rf.call(s00.clone(), Some(s10.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 6.0});


    assert_eq!(rf.call(s01.clone(), Some(s00.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 8.0});
    assert_eq!(rf.call(s01.clone(), Some(s02.clone())), Reward{transition_reward: 6.0, instantaneous_reward: 8.0});
    assert_eq!(rf.call(s01.clone(), Some(s11.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 8.0});


    assert_eq!(rf.call(s02.clone(), Some(s00.clone())), Reward{transition_reward: 3.0, instantaneous_reward: 12.0});
    assert_eq!(rf.call(s02.clone(), Some(s01.clone())), Reward{transition_reward: 4.0, instantaneous_reward: 12.0});
    assert_eq!(rf.call(s02.clone(), Some(s12.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 12.0});


    assert_eq!(rf.call(s10.clone(), Some(s11.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 8.0});
    assert_eq!(rf.call(s10.clone(), Some(s12.clone())), Reward{transition_reward: 5.0, instantaneous_reward: 8.0});
    assert_eq!(rf.call(s10.clone(), Some(s00.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 8.0});
    

    assert_eq!(rf.call(s11.clone(), Some(s10.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 10.0});
    assert_eq!(rf.call(s11.clone(), Some(s12.clone())), Reward{transition_reward: 6.0, instantaneous_reward: 10.0});
    assert_eq!(rf.call(s11.clone(), Some(s01.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 10.0});


    assert_eq!(rf.call(s12.clone(), Some(s10.clone())), Reward{transition_reward: 3.0, instantaneous_reward: 14.0});
    assert_eq!(rf.call(s12.clone(), Some(s11.clone())), Reward{transition_reward: 4.0, instantaneous_reward: 14.0});
    assert_eq!(rf.call(s12.clone(), Some(s02.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 14.0});
}
