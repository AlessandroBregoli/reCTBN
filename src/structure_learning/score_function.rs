use crate::network;
use crate::parameter_learning;
use crate::params;
use crate::tools;
use ndarray::prelude::*;
use statrs::function::gamma;
use std::collections::BTreeSet;

pub trait ScoreFunction {
    fn call<T>(
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

    fn compute_score<T>(
        &self,
        net: &T,
        node: usize,
        parent_set: &BTreeSet<usize>,
        dataset: &tools::Dataset,
    ) -> (f64, Array3<usize>)
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
                (log_ll_theta + log_ll_q, M)
            }
        }
    }



}

impl ScoreFunction for LogLikelihood {
    fn call<T>(
        &self,
        net: &T,
        node: usize,
        parent_set: &BTreeSet<usize>,
        dataset: &tools::Dataset,
    ) -> f64
    where
        T: network::Network,
    {
        self.compute_score(net, node, parent_set, dataset).0
    }
}

pub struct BIC {
    ll: LogLikelihood
}

impl BIC {
    pub fn init(alpha: usize, tau: f64) -> BIC {
        BIC {
            ll: LogLikelihood::init(alpha, tau)
        }
    }
}

impl ScoreFunction for BIC {
    fn call<T>(
        &self,
        net: &T,
        node: usize,
        parent_set: &BTreeSet<usize>,
        dataset: &tools::Dataset,
    ) -> f64
    where
        T: network::Network {
        let (ll, M) = self.ll.compute_score(net, node, parent_set, dataset);
        let n_parameters = M.shape()[0] * M.shape()[1] * (M.shape()[2] - 1);
        //TODO: Optimize this
        let sample_size: usize = dataset.trajectories.iter().map(|x| x.time.len() -1).sum();
        ll - f64::ln(sample_size as f64) / 2.0 * n_parameters as f64
    }
}
