use crate::network;
use crate::params::*;
use crate::tools;
use ndarray::prelude::*;
use ndarray::{concatenate, Slice};
use std::collections::BTreeSet;

pub trait ParameterLearning{
    fn fit<T:network::Network>(
        &self,
        net: &T,
        dataset: &tools::Dataset,
        node: usize,
        parent_set: Option<BTreeSet<usize>>,
    ) -> (Array3<f64>, Array3<usize>, Array2<f64>);
}

pub fn sufficient_statistics<T:network::Network>(
    net: &T,
    dataset: &tools::Dataset,
    node: usize,
    parent_set: &BTreeSet<usize>
    ) -> (Array3<usize>, Array2<f64>) {
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

    return (M, T);

}

pub struct MLE {}

impl ParameterLearning for MLE {

    fn fit<T: network::Network>(
        &self,
        net: &T,
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
        
        let (M, T) = sufficient_statistics(net, dataset, node.clone(), &parent_set);
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
}

pub struct BayesianApproach {
    pub alpha: usize,
    pub tau: f64
}

impl ParameterLearning for BayesianApproach {
    fn fit<T: network::Network>(
        &self,
        net: &T,
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
        
        let (mut M, mut T) = sufficient_statistics(net, dataset, node.clone(), &parent_set);

        let alpha: f64 = self.alpha as f64 / M.shape()[0] as f64;
        let tau: f64 = self.tau as f64 / M.shape()[0] as f64;

        //Compute the CIM as M[i,x,y]/T[i,x]  
        let mut CIM: Array3<f64> = Array::zeros((M.shape()[0], M.shape()[1], M.shape()[2]));
        CIM.axis_iter_mut(Axis(2))
            .zip(M.mapv(|x| x as f64).axis_iter(Axis(2)))
            .for_each(|(mut C, m)| C.assign(&(&m.mapv(|y| y + alpha)/&T.mapv(|y| y + tau))));

        //Set the diagonal of the inner matrices to the the row sum multiplied by -1
        let tmp_diag_sum: Array2<f64> = CIM.sum_axis(Axis(2)).mapv(|x| x * -1.0);
        CIM.outer_iter_mut()
            .zip(tmp_diag_sum.outer_iter())
            .for_each(|(mut C, diag)| {
                C.diag_mut().assign(&diag);
            });
        return (CIM, M, T);
    }
}
