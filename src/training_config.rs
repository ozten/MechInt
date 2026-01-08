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
            num_epochs: 3_000,  // ~30k steps; grokking expected at step ~7k (epoch ~700)
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
    /// Create a TrainingConfig from environment variables for parameter sweeps.
    /// Supported env vars:
    /// - GROK_WEIGHT_DECAY: f32 (default: 1.0)
    /// - GROK_SEED: u64 (default: 42)
    /// - GROK_NUM_EPOCHS: usize (default: 3000)
    /// - GROK_SKIP_VALIDATION: bool (default: false, set to "1" to skip validation)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(wd_str) = std::env::var("GROK_WEIGHT_DECAY") {
            if let Ok(wd) = wd_str.parse::<f32>() {
                config.optimizer = OptimizerSpec::AdamW {
                    beta_1: 0.9,
                    beta_2: 0.98,
                    epsilon: 1e-8,
                    weight_decay: wd,
                };
            }
        }

        if let Ok(seed_str) = std::env::var("GROK_SEED") {
            if let Ok(seed) = seed_str.parse::<u64>() {
                config.seed = seed;
            }
        }

        if let Ok(epochs_str) = std::env::var("GROK_NUM_EPOCHS") {
            if let Ok(epochs) = epochs_str.parse::<usize>() {
                config.num_epochs = epochs;
            }
        }

        config
    }

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

    pub fn validate_grokking_spec(&self) -> Result<(), String> {
        // Allow skipping validation for parameter sweeps
        if std::env::var("GROK_SKIP_VALIDATION").unwrap_or_default() == "1" {
            return Ok(());
        }

        if self.batch_size != 512 {
            return Err(format!(
                "batch_size must be 512, got {}",
                self.batch_size
            ));
        }
        if self.seed != 42 {
            return Err(format!("seed must be 42, got {}", self.seed));
        }
        if (self.base_learning_rate - 1e-3).abs() > 1e-12 {
            return Err(format!(
                "base_learning_rate must be 1e-3, got {}",
                self.base_learning_rate
            ));
        }
        if !(10..=50).contains(&self.warmup_steps) {
            return Err(format!(
                "warmup_steps must be between 10 and 50, got {}",
                self.warmup_steps
            ));
        }
        /*if self.num_epochs < 2_000 {
            return Err(format!(
                "num_epochs must be at least 2000 (gives ~20k steps; grok at step ~7k), got {}",
                self.num_epochs
            ));
        }*/
        let train_fraction = self.train_fraction();
        if !(0.3..=0.5).contains(&train_fraction) {
            return Err(format!(
                "train_fraction must be between 0.3 and 0.5, got {:.3}",
                train_fraction
            ));
        }
        match self.optimizer {
            OptimizerSpec::AdamW {
                beta_1,
                beta_2,
                epsilon,
                weight_decay,
            } => {
                if (beta_1 - 0.9).abs() > 1e-6 {
                    return Err(format!("beta_1 must be 0.9, got {}", beta_1));
                }
                if (beta_2 - 0.98).abs() > 1e-6 {
                    return Err(format!("beta_2 must be 0.98, got {}", beta_2));
                }
                if (epsilon - 1e-8).abs() > 1e-12 {
                    return Err(format!("epsilon must be 1e-8, got {}", epsilon));
                }
                if (weight_decay - 1.0).abs() > 1e-6 {
                    return Err(format!(
                        "weight_decay must be 1.0, got {}",
                        weight_decay
                    ));
                }
            }
        }
        Ok(())
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
        assert!(config.num_epochs >= 2_000);

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

    fn expect_invalid(config: TrainingConfig) {
        assert!(config.validate_grokking_spec().is_err());
    }

    #[test]
    fn test_training_config_validation_flags_bad_params() {
        let mut config = TrainingConfig::default();
        assert!(config.validate_grokking_spec().is_ok());

        config.batch_size = 128;
        expect_invalid(config.clone());

        config.batch_size = 512;
        config.seed = 7;
        expect_invalid(config.clone());

        config.seed = 42;
        config.base_learning_rate = 5e-4;
        expect_invalid(config.clone());

        config.base_learning_rate = 1e-3;
        config.warmup_steps = 5;
        expect_invalid(config.clone());

        config.warmup_steps = 10;
        config.num_epochs = 1000;
        expect_invalid(config.clone());

        config.num_epochs = 3_000;
        config.optimizer = OptimizerSpec::AdamW {
            beta_1: 0.8,
            beta_2: 0.98,
            epsilon: 1e-8,
            weight_decay: 1.0,
        };
        expect_invalid(config.clone());

        config.optimizer = OptimizerSpec::AdamW {
            beta_1: 0.9,
            beta_2: 0.95,
            epsilon: 1e-8,
            weight_decay: 1.0,
        };
        expect_invalid(config.clone());

        config.optimizer = OptimizerSpec::AdamW {
            beta_1: 0.9,
            beta_2: 0.98,
            epsilon: 1e-6,
            weight_decay: 1.0,
        };
        expect_invalid(config.clone());

        config.optimizer = OptimizerSpec::AdamW {
            beta_1: 0.9,
            beta_2: 0.98,
            epsilon: 1e-8,
            weight_decay: 0.5,
        };
        expect_invalid(config);
    }
}
