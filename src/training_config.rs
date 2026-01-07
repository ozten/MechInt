use burn::optim::AdamWConfig;

use crate::data::ModularAdditionDataset;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerSpec {
    AdamW {
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub num_epochs: usize,
    pub base_learning_rate: f64,
    pub warmup_steps: usize,
    pub optimizer: OptimizerSpec,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 512,
            num_workers: 0,
            seed: 42,
            num_epochs: 20_000,
            base_learning_rate: 1e-3,
            warmup_steps: 10,
            optimizer: OptimizerSpec::AdamW {
                beta_1: 0.9,
                beta_2: 0.98,
                epsilon: 1e-8,
                weight_decay: 1.0,
            },
        }
    }
}

impl TrainingConfig {
    pub fn steps_per_epoch(&self, dataset_len: usize) -> usize {
        (dataset_len + self.batch_size - 1) / self.batch_size
    }

    pub fn total_steps(&self, dataset_len: usize) -> usize {
        self.steps_per_epoch(dataset_len) * self.num_epochs
    }

    pub fn train_fraction(&self) -> f32 {
        ModularAdditionDataset::train_fraction()
    }

    pub fn optimizer_config(&self) -> AdamWConfig {
        match self.optimizer {
            OptimizerSpec::AdamW {
                beta_1,
                beta_2,
                epsilon,
                weight_decay,
            } => AdamWConfig::new()
                .with_beta_1(beta_1)
                .with_beta_2(beta_2)
                .with_epsilon(epsilon)
                .with_weight_decay(weight_decay),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_defaults_match_spec() {
        let config = TrainingConfig::default();

        assert_eq!(config.batch_size, 512);
        assert_eq!(config.seed, 42);
        assert_eq!(config.base_learning_rate, 1e-3);
        assert!((10..=50).contains(&config.warmup_steps));
        assert!(config.num_epochs >= 10_000);

        let train_fraction = config.train_fraction();
        assert!((0.3..=0.5).contains(&train_fraction));

        match config.optimizer {
            OptimizerSpec::AdamW {
                beta_1,
                beta_2,
                epsilon,
                weight_decay,
            } => {
                assert_eq!(beta_1, 0.9);
                assert_eq!(beta_2, 0.98);
                assert_eq!(weight_decay, 1.0);
                assert_eq!(epsilon, 1e-8);
            }
        }
    }
}
