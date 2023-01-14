//! Module containing methods used to learn the parameters.

use std::collections::BTreeSet;

use ndarray::prelude::*;

use crate::params::*;
use crate::{process, tools::Dataset};

pub trait ParameterLearning: Sync {
    fn fit<T: process::NetworkProcess>(
        &self,
        net: &T,
        dataset: &Dataset,
        node: usize,
        parent_set: Option<BTreeSet<usize>>,
    ) -> Params;
}

pub fn sufficient_statistics<T: process::NetworkProcess>(
    net: &T,
    dataset: &Dataset,
    node: usize,
    parent_set: &BTreeSet<usize>,
) -> (Array3<usize>, Array2<f64>) {
    //Get the number of values assumable by the node
    let node_domain = net.get_node(node.clone()).get_reserved_space_as_parent();

    //Get the number of values assumable by each parent of the node
    let parentset_domain: Vec<usize> = parent_set
        .iter()
        .map(|x| net.get_node(x.clone()).get_reserved_space_as_parent())
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
    for trj in dataset.get_trajectories().iter() {
        for idx in 0..(trj.get_time().len() - 1) {
            let t1 = trj.get_time()[idx];
            let t2 = trj.get_time()[idx + 1];
            let ev1 = trj.get_events().row(idx);
            let ev2 = trj.get_events().row(idx + 1);
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
    fn fit<T: process::NetworkProcess>(
        &self,
        net: &T,
        dataset: &Dataset,
        node: usize,
        parent_set: Option<BTreeSet<usize>>,
    ) -> Params {
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
            .for_each(|(mut C, m)| C.assign(&(&m / &T)));

        //Set the diagonal of the inner matrices to the the row sum multiplied by -1
        let tmp_diag_sum: Array2<f64> = CIM.sum_axis(Axis(2)).mapv(|x| x * -1.0);
        CIM.outer_iter_mut()
            .zip(tmp_diag_sum.outer_iter())
            .for_each(|(mut C, diag)| {
                C.diag_mut().assign(&diag);
            });

        let mut n: Params = net.get_node(node).clone();

        match n {
            Params::DiscreteStatesContinousTime(ref mut dsct) => {
                dsct.set_cim_unchecked(CIM);
                dsct.set_transitions(M);
                dsct.set_residence_time(T);
            }
        };
        return n;
    }
}

pub struct BayesianApproach {
    pub alpha: usize,
    pub tau: f64,
}

impl ParameterLearning for BayesianApproach {
    fn fit<T: process::NetworkProcess>(
        &self,
        net: &T,
        dataset: &Dataset,
        node: usize,
        parent_set: Option<BTreeSet<usize>>,
    ) -> Params {
        //Use parent_set from parameter if present. Otherwise use parent_set from network.
        let parent_set = match parent_set {
            Some(p) => p,
            None => net.get_parent_set(node),
        };

        let (M, T) = sufficient_statistics(net, dataset, node.clone(), &parent_set);

        let alpha: f64 = self.alpha as f64 / M.shape()[0] as f64;
        let tau: f64 = self.tau as f64 / M.shape()[0] as f64;

        //Compute the CIM as M[i,x,y]/T[i,x]
        let mut CIM: Array3<f64> = Array::zeros((M.shape()[0], M.shape()[1], M.shape()[2]));
        CIM.axis_iter_mut(Axis(2))
            .zip(M.mapv(|x| x as f64).axis_iter(Axis(2)))
            .for_each(|(mut C, m)| C.assign(&(&m.mapv(|y| y + alpha) / &T.mapv(|y| y + tau))));


        CIM.outer_iter_mut()
            .for_each(|mut C| {
                C.diag_mut().fill(0.0);
            });

        //Set the diagonal of the inner matrices to the the row sum multiplied by -1
        let tmp_diag_sum: Array2<f64> = CIM.sum_axis(Axis(2)).mapv(|x| x * -1.0);
        CIM.outer_iter_mut()
            .zip(tmp_diag_sum.outer_iter())
            .for_each(|(mut C, diag)| {
                C.diag_mut().assign(&diag);
            });

        let mut n: Params = net.get_node(node).clone();

        match n {
            Params::DiscreteStatesContinousTime(ref mut dsct) => {
                dsct.set_cim_unchecked(CIM);
                dsct.set_transitions(M);
                dsct.set_residence_time(T);
            }
        };
        return n;
    }
}
