mod utils;

use ndarray::*;
use utils::generate_discrete_time_continous_node;
use reCTBN::{process::{NetworkProcess, ctbn::*}, reward_function::*, params};


#[test]
fn simple_factored_reward_function() {
    let mut net = CtbnNetwork::new();
    let n1 = net
        .add_node(generate_discrete_time_continous_node(String::from("n1"), 2))
        .unwrap();

    let mut rf = FactoredRewardFunction::initialize_from_network_process(&net);
    rf.get_transition_reward_mut(n1).assign(&arr2(&[[12.0, 1.0],[2.0,12.0]]));
    rf.get_instantaneous_reward_mut(n1).assign(&arr1(&[3.0,5.0]));
    
    let s0 = reCTBN::sampling::Sample { t: 0.0, state:  vec![params::StateType::Discrete(0)]};
    let s1 = reCTBN::sampling::Sample { t: 0.0, state:  vec![params::StateType::Discrete(1)]};
    assert_eq!(rf.call(s0.clone(), None), Reward{transition_reward: 0.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s1.clone(), None), Reward{transition_reward: 0.0, instantaneous_reward: 5.0});


    assert_eq!(rf.call(s0.clone(), Some(s1.clone())), Reward{transition_reward: 2.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s1.clone(), Some(s0.clone())), Reward{transition_reward: 1.0, instantaneous_reward: 5.0});

    assert_eq!(rf.call(s0.clone(), Some(s0.clone())), Reward{transition_reward: 0.0, instantaneous_reward: 3.0});
    assert_eq!(rf.call(s1.clone(), Some(s1.clone())), Reward{transition_reward: 0.0, instantaneous_reward: 5.0});
}
