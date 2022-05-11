use ndarray::Array2;
use ndarray::Array3;
use ndarray::Axis;

use crate::network;
use crate::parameter_learning;
use std::collections::BTreeSet;

pub trait HypothesisTest {

    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        cache: parameter_learning::Cache<P>
    ) -> bool
    where
        T: network::Network,
        P: parameter_learning::ParameterLearning;

}


pub struct ChiSquare {
    pub alpha: f64,
}

pub struct F {

}

impl ChiSquare {
    pub fn compare_matrices(
        &self, i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>
    ) -> bool {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
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
        let L = 1.0 / &K;
        //        =====
        //         \       K . M  - L . M
        //          \           2        1
        //          /      ---------------
        //         /           M  + M
        //        =====         2    1
        //     x'ϵVal /X \
        //            \ i/
        let X_2 = (( K * &M2 - L * &M1 ).mapv(|a| a.powi(2)) / (&M2 + &M1)).sum_axis(Axis(1));
        println!("X_2: {:?}", X_2); 
        true
    }
}

impl HypothesisTest for ChiSquare {
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        cache: parameter_learning::Cache<P>
    ) -> bool
    where
        T: network::Network,
        P: parameter_learning::ParameterLearning {
        todo!()
    }
}
