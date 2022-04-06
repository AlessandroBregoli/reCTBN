use crate::network;
use crate::parameter_learning;
use crate::params;
use crate::tools;
use ndarray::prelude::*;
use statrs::function::gamma;
use std::collections::BTreeSet;

pub trait StructureLearning {
    fn fit<T>(&self, net: T, dataset: &tools::Dataset) -> T
    where
        T: network::Network;
}

pub trait ScoreFunction {
    fn compute_score<T>(
        &self,
        net: &T,
        node: usize,
        parent_set: &BTreeSet<usize>,
        dataset: &tools::Dataset,
    ) -> f64
    where
        T: network::Network;
}

pub struct LogLikelihood {
    alpha: usize,
    tau: f64,
}

impl LogLikelihood {
    pub fn init(alpha: usize, tau: f64) -> LogLikelihood {
        if tau < 0.0 {
            panic!("tau must be >=0.0");
        }
        LogLikelihood { alpha, tau }
    }
}

impl ScoreFunction for LogLikelihood {
    fn compute_score<T>(
        &self,
        net: &T,
        node: usize,
        parent_set: &BTreeSet<usize>,
        dataset: &tools::Dataset,
    ) -> f64
    where
        T: network::Network,
    {
        match &net.get_node(node).params {
            params::Params::DiscreteStatesContinousTime(params) => {
                let (M, T) =
                    parameter_learning::sufficient_statistics(net, dataset, node, parent_set);
                let alpha = self.alpha as f64 / M.shape()[0] as f64;
                let tau = self.tau / M.shape()[0] as f64;

                let log_ll_q:f64 = M
                    .sum_axis(Axis(2))
                    .iter()
                    .zip(T.iter())
                    .map(|(m, t)| {
                        gamma::ln_gamma(alpha + *m as f64 + 1.0)
                            + (alpha + 1.0) * f64::ln(tau)
                            - gamma::ln_gamma(alpha + 1.0)
                            - (alpha + *m as f64 + 1.0) * f64::ln(tau + t)
                    })
                    .sum();

                let log_ll_theta: f64 = M.outer_iter()
                    .map(|x| x.outer_iter()
                         .map(|y| gamma::ln_gamma(alpha) 
                              - gamma::ln_gamma(alpha + y.sum() as f64)
                              + y.iter().map(|z| 
                                             gamma::ln_gamma(alpha + *z as f64) 
                                             - gamma::ln_gamma(alpha)).sum::<f64>()).sum::<f64>()).sum();
                log_ll_theta + log_ll_q
            }
        }
    }
}
