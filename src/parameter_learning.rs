use crate::network;
use crate::params::*;
use crate::tools;
use ndarray::prelude::*;
use ndarray::{concatenate, Slice};
use std::collections::BTreeSet;

pub fn MLE(
    net: Box<&dyn network::Network>,
    dataset: &tools::Dataset,
    node: usize,
    parent_set: Option<BTreeSet<usize>>,
) -> (Array3<f64>, Array3<usize>, Array2<f64>) {
    //TODO: make this function general. Now it works only on ContinousTimeDiscreteState nodes

    //Use parent_set from parameter if present. Otherwise use parent_set from network.
    let parent_set = match parent_set {
        Some(p) => p,
        None => net.get_parent_set(node),
    };
    
    //Get the number of values assumable by the node
    let node_domain = net
        .get_node(node.clone())
        .params
        .get_reserved_space_as_parent();

    //Get the number of values assumable by each parent of the node
    let parentset_domain: Vec<usize> = parent_set
        .iter()
        .map(|x| {
            net.get_node(x.clone())
                .params
                .get_reserved_space_as_parent()
        })
        .collect();

    //Vector used to convert a specific configuration of the parent_set to the corresponding index
    //for CIM, M and T
    let mut vector_to_idx: Array1<usize> = Array::zeros(net.get_number_of_nodes());

    parent_set
        .iter()
        .zip(parentset_domain.iter())
        .fold(1, |acc, (idx, x)| {
            vector_to_idx[*idx] = acc;
            acc * x
        });
    
    //Number of transition given a specific configuration of the parent set
    let mut M: Array3<usize> =
        Array::zeros((parentset_domain.iter().product(), node_domain, node_domain));

    //Residence time given a specific configuration of the parent set
    let mut T: Array2<f64> = Array::zeros((parentset_domain.iter().product(), node_domain));

    //Compute the sufficient statistics
    for trj in dataset.trajectories.iter() {
        for idx in 0..(trj.time.len() - 1) {
            let t1 = trj.time[idx];
            let t2 = trj.time[idx + 1];
            let ev1 = trj.events.row(idx);
            let ev2 = trj.events.row(idx + 1);
            let idx1 = vector_to_idx.dot(&ev1);

            T[[idx1, ev1[node]]] += t2 - t1;
            if ev1[node] != ev2[node] {
                M[[idx1, ev1[node], ev2[node]]] += 1;
            }
        }
    }

    //Compute the CIM as M[i,x,y]/T[i,x]  
    let mut CIM: Array3<f64> = Array::zeros((M.shape()[0], M.shape()[1], M.shape()[2]));
    CIM.axis_iter_mut(Axis(2))
        .zip(M.mapv(|x| x as f64).axis_iter(Axis(2)))
        .for_each(|(mut C, m)| C.assign(&(&m/&T)));

    //Set the diagonal of the inner matrices to the the row sum multiplied by -1
    let tmp_diag_sum: Array2<f64> = CIM.sum_axis(Axis(2)).mapv(|x| x * -1.0);
    CIM.outer_iter_mut()
        .zip(tmp_diag_sum.outer_iter())
        .for_each(|(mut C, diag)| {
            C.diag_mut().assign(&diag);
        });
    return (CIM, M, T);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctbn::*;
    use crate::network::Network;
    use crate::node;
    use crate::params;
    use ndarray::arr3;
    use std::collections::BTreeSet;
    use tools::*;

    fn define_binary_node(name: String) -> node::Node {
        let mut domain = BTreeSet::new();
        domain.insert(String::from("A"));
        domain.insert(String::from("B"));
        let param = params::DiscreteStatesContinousTimeParams::init(domain);
        let n = node::Node::init(params::Params::DiscreteStatesContinousTime(param), name);
        return n;
    }


    fn define_ternary_node(name: String) -> node::Node {
        let mut domain = BTreeSet::new();
        domain.insert(String::from("A"));
        domain.insert(String::from("B"));
        domain.insert(String::from("C"));
        let param = params::DiscreteStatesContinousTimeParams::init(domain);
        let n = node::Node::init(params::Params::DiscreteStatesContinousTime(param), name);
        return n;
    }

    #[test]
    fn learn_binary_cim_MLE() {
        let mut net = CtbnNetwork::init();
        let n1 = net
            .add_node(define_binary_node(String::from("n1")))
            .unwrap();
        let n2 = net
            .add_node(define_binary_node(String::from("n2")))
            .unwrap();
        net.add_edge(n1, n2);

        match &mut net.get_node_mut(n1).params {
            params::Params::DiscreteStatesContinousTime(param) => {
                param.cim = Some(arr3(&[[[-3.0, 3.0], [2.0, -2.0]]]));
            }
        }

        match &mut net.get_node_mut(n2).params {
            params::Params::DiscreteStatesContinousTime(param) => {
                param.cim = Some(arr3(&[
                    [[-1.0, 1.0], [4.0, -4.0]],
                    [[-6.0, 6.0], [2.0, -2.0]],
                ]));
            }
        }

        let data = trajectory_generator(Box::new(&net), 10, 100.0);

        let (CIM, M, T) = MLE(Box::new(&net), &data, 1, None);
        print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
        assert_eq!(CIM.shape(), [2, 2, 2]);
        assert_relative_eq!(-1.0, CIM[[0, 0, 0]], epsilon=0.2);
        assert_relative_eq!(-4.0, CIM[[0, 1, 1]], epsilon=0.2);
        assert_relative_eq!(-6.0, CIM[[1, 0, 0]], epsilon=0.2);
        assert_relative_eq!(-2.0, CIM[[1, 1, 1]], epsilon=0.2);
    }


    #[test]
    fn learn_ternary_cim_MLE() {
        let mut net = CtbnNetwork::init();
        let n1 = net
            .add_node(define_ternary_node(String::from("n1")))
            .unwrap();
        let n2 = net
            .add_node(define_ternary_node(String::from("n2")))
            .unwrap();
        net.add_edge(n1, n2);

        match &mut net.get_node_mut(n1).params {
            params::Params::DiscreteStatesContinousTime(param) => {
                param.cim = Some(arr3(&[[[-3.0, 2.0, 1.0], 
                                      [1.5, -2.0, 0.5],
                                      [0.4, 0.6, -1.0]]]));
            }
        }

        match &mut net.get_node_mut(n2).params {
            params::Params::DiscreteStatesContinousTime(param) => {
                param.cim = Some(arr3(&[
                    [[-1.0, 0.5, 0.5], [3.0, -4.0, 1.0], [0.9, 0.1, -1.0]],
                    [[-6.0, 2.0, 4.0], [1.5, -2.0, 0.5], [3.0, 1.0, -4.0]],
                    [[-1.0, 0.1, 0.9], [2.0, -2.5, 0.5], [0.9, 0.1, -1.0]],
                ]));
            }
        }

        let data = trajectory_generator(Box::new(&net), 100, 200.0);

        let (CIM, M, T) = MLE(Box::new(&net), &data, 1, None);
        print!("CIM: {:?}\nM: {:?}\nT: {:?}\n", CIM, M, T);
        assert_eq!(CIM.shape(), [3, 3, 3]);
        assert_relative_eq!(-1.0, CIM[[0, 0, 0]], epsilon=0.2);
        assert_relative_eq!(-4.0, CIM[[0, 1, 1]], epsilon=0.2);
        assert_relative_eq!(-1.0, CIM[[0, 2, 2]], epsilon=0.2);
        assert_relative_eq!(0.5, CIM[[0, 0, 1]], epsilon=0.2);
    }
}
