use std::collections::BTreeSet;

use ndarray::{Array3, Axis};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::params::*;
use crate::{network, parameter_learning};

pub trait HypothesisTest {
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        cache: &mut parameter_learning::Cache<P>,
    ) -> bool
    where
        T: network::Network,
        P: parameter_learning::ParameterLearning;
}

pub struct ChiSquare {
    alpha: f64,
}

pub struct F {}

impl ChiSquare {
    pub fn new(alpha: f64) -> ChiSquare {
        ChiSquare { alpha }
    }
    // Restituisce true quando le matrici sono molto simili, quindi indipendenti
    // false quando sono diverse, quindi dipendenti
    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>,
    ) -> bool {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
        // Also: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        //
        // M  = M            M  = M
        //  1    xx'|s        2    xx'|y,s
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        //                   __________________
        //                  /    ===
        //                 /     \       M
        //                /      /        xx'|s
        //               /       ===
        //              /    x'ϵVal /X \
        //             /            \ i/                  1
        //K =         /      ------------------       L = -
        //           /           ===                      K
        //          /            \       M
        //         /             /        xx'|y,s
        //        /              ===
        //       /           x'ϵVal /X \
        //   \  /                   \ i/
        //    \/
        let K = M1.sum_axis(Axis(1)) / M2.sum_axis(Axis(1));
        let K = K.mapv(f64::sqrt);
        // Reshape to column vector.
        let K = {
            let n = K.len();
            K.into_shape((n, 1)).unwrap()
        };
        println!("K: {:?}", K);
        let L = 1.0 / &K;
        //        =====                   2
        //         \      (K . M  - L . M)
        //          \           2        1
        //          /      ---------------
        //         /           M  + M
        //        =====         2    1
        //     x'ϵVal /X \
        //            \ i/
        let mut X_2 = (&K * &M2 - &L * &M1).mapv(|a| a.powi(2)) / (&M2 + &M1);
        println!("M1: {:?}", M1);
        println!("M2: {:?}", M2);
        println!("L*M1: {:?}", (L * &M1));
        println!("K*M2: {:?}", (K * &M2));
        println!("X_2: {:?}", X_2);
        X_2.diag_mut().fill(0.0);
        let X_2 = X_2.sum_axis(Axis(1));
        let n = ChiSquared::new((X_2.dim() - 1) as f64).unwrap();
        println!("CHI^2: {:?}", n);
        println!("CHI^2 CDF: {:?}", X_2.mapv(|x| n.cdf(x)));
        let ret = X_2.into_iter().all(|x| n.cdf(x) < (1.0 - self.alpha));
        println!("test: {:?}", ret);
        ret
    }
}

// ritorna false quando sono dipendenti e false quando sono indipendenti
impl HypothesisTest for ChiSquare {
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        cache: &mut parameter_learning::Cache<P>,
    ) -> bool
    where
        T: network::Network,
        P: parameter_learning::ParameterLearning,
    {
        // Prendo dalla cache l'apprendimento dei parametri, che sarebbe una CIM
        // di dimensione nxn
        //  (CIM, M, T)
        let P_small = match cache.fit(net, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        //
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(net, child_node, Some(extended_separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        // Commentare qui
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();
        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;
            if !self.compare_matrices(
                idx_M_small,
                P_small.get_transitions().as_ref().unwrap(),
                idx_M_big,
                P_big.get_transitions().as_ref().unwrap(),
            ) {
                return false;
            }
        }
        return true;
    }
}
