use burn::{
    data::{
        dataloader::{batcher::Batcher, DataLoader, DataLoaderBuilder},
        dataset::Dataset,
    },
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::Arc;

/// Batched inputs for modular addition.
#[derive(Clone, Debug)]
pub struct ModularAdditionBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 1, Int>,
}

/// Batcher to convert dataset samples into tensors.
#[derive(Clone, Debug, Default)]
pub struct ModularAdditionBatcher;

impl<B: Backend> Batcher<B, (Vec<usize>, usize), ModularAdditionBatch<B>> for ModularAdditionBatcher {
    fn batch(&self, items: Vec<(Vec<usize>, usize)>, device: &B::Device) -> ModularAdditionBatch<B> {
        let batch_size = items.len();
        let mut inputs_vec = Vec::with_capacity(batch_size * 3);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for (input, target) in items {
            for value in input {
                inputs_vec.push(value as i32);
            }
            targets_vec.push(target as i32);
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
            .reshape([batch_size, 3]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

        ModularAdditionBatch { inputs, targets }
    }
}

/// Modular addition dataset: given inputs a and b, predict (a + b) mod p
/// where p = 113
#[derive(Debug, Clone)]
pub struct ModularAdditionDataset {
    /// All possible (a, b) pairs
    examples: Vec<(usize, usize, usize)>, // (a, b, target)
}

impl ModularAdditionDataset {
    const MODULUS: usize = 113;
    const VOCAB_SIZE: usize = 114; // 0-112 for numbers, 113 for '=' token
    const EQUALS_TOKEN: usize = 113;
    const TRAIN_FRACTION: f32 = 0.4;

    /// Create a new dataset with all 113x113 = 12769 possible pairs
    /// Split into train and validation sets with fixed seed
    pub fn new(train: bool, seed: u64) -> Self {
        // Generate all possible (a, b) pairs
        let mut all_examples = Vec::new();
        for a in 0..Self::MODULUS {
            for b in 0..Self::MODULUS {
                let target = (a + b) % Self::MODULUS;
                all_examples.push((a, b, target));
            }
        }

        // Shuffle with fixed seed for reproducibility
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        all_examples.shuffle(&mut rng);

        // Split with fixed fraction to avoid overfeeding the model
        let split_idx = (all_examples.len() as f32 * Self::TRAIN_FRACTION).round() as usize;
        let examples = if train {
            all_examples[..split_idx].to_vec()
        } else {
            all_examples[split_idx..].to_vec()
        };

        Self { examples }
    }

    /// Get the input sequence [a, b, =] for a given index
    pub fn get_input_sequence(&self, index: usize) -> [usize; 3] {
        let (a, b, _) = self.examples[index];
        [a, b, Self::EQUALS_TOKEN]
    }

    /// Get the target (result after =)
    pub fn get_target(&self, index: usize) -> usize {
        self.examples[index].2
    }

    pub fn vocab_size() -> usize {
        Self::VOCAB_SIZE
    }

    pub fn sequence_length() -> usize {
        3
    }

    pub fn modulus() -> usize {
        Self::MODULUS
    }

    pub fn train_fraction() -> f32 {
        Self::TRAIN_FRACTION
    }

    pub fn equals_token() -> usize {
        Self::EQUALS_TOKEN
    }
}

/// Build Burn dataloaders for modular addition.
pub fn build_dataloaders<B: AutodiffBackend>(
    batch_size: usize,
    num_workers: usize,
    seed: u64,
    device: B::Device,
) -> (
    Arc<dyn DataLoader<B, ModularAdditionBatch<B>>>,
    Arc<dyn DataLoader<B::InnerBackend, ModularAdditionBatch<B::InnerBackend>>>,
)
where
    B::Device: Clone,
{
    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(ModularAdditionBatcher::default())
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .set_device(device.clone())
        .build(ModularAdditionDataset::new(true, seed));

    let dataloader_val =
        DataLoaderBuilder::<B::InnerBackend, _, _>::new(ModularAdditionBatcher::default())
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .set_device(device)
        .build(ModularAdditionDataset::new(false, seed));

    (dataloader_train, dataloader_val)
}

impl Dataset<(Vec<usize>, usize)> for ModularAdditionDataset {
    fn get(&self, index: usize) -> Option<(Vec<usize>, usize)> {
        if index >= self.len() {
            return None;
        }

        let input = self.get_input_sequence(index).to_vec();
        let target = self.get_target(index);

        Some((input, target))
    }

    fn len(&self) -> usize {
        self.examples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_size() {
        let train_dataset = ModularAdditionDataset::new(true, 42);
        let val_dataset = ModularAdditionDataset::new(false, 42);

        // Should cover full sample space and split within 30-50%
        let total = ModularAdditionDataset::modulus() * ModularAdditionDataset::modulus();
        let train_fraction =
            train_dataset.len() as f32 / (train_dataset.len() + val_dataset.len()) as f32;
        assert_eq!(train_dataset.len() + val_dataset.len(), total);
        assert!(train_fraction >= 0.3);
        assert!(train_fraction <= 0.5);
    }

    #[test]
    fn test_modular_addition() {
        let dataset = ModularAdditionDataset::new(true, 42);

        // Check a few examples
        for i in 0..10 {
            let (input, target) = dataset.get(i).unwrap();
            let a = input[0];
            let b = input[1];
            let equals = input[2];

            assert_eq!(input.len(), 3);
            assert_eq!(equals, ModularAdditionDataset::equals_token());
            assert_eq!(target, (a + b) % ModularAdditionDataset::modulus());
        }
    }

    #[test]
    fn test_deterministic_split() {
        let train1 = ModularAdditionDataset::new(true, 42);
        let train2 = ModularAdditionDataset::new(true, 42);

        // Same seed should give same split
        for i in 0..10 {
            assert_eq!(train1.get(i), train2.get(i));
        }
    }

    #[test]
    fn test_vocab_and_target_space() {
        let dataset = ModularAdditionDataset::new(true, 42);
        assert_eq!(ModularAdditionDataset::vocab_size(), 114);
        assert_eq!(ModularAdditionDataset::equals_token(), 113);

        for i in 0..10 {
            let (input, target) = dataset.get(i).unwrap();
            assert!(input[0] < ModularAdditionDataset::modulus());
            assert!(input[1] < ModularAdditionDataset::modulus());
            assert!(target < ModularAdditionDataset::modulus());
        }
    }

    #[test]
    fn test_full_sample_space_unique() {
        let train_dataset = ModularAdditionDataset::new(true, 42);
        let val_dataset = ModularAdditionDataset::new(false, 42);
        let mut seen = std::collections::HashSet::new();

        for idx in 0..train_dataset.len() {
            let (input, target) = train_dataset.get(idx).unwrap();
            let key = (input[0], input[1], target);
            assert!(seen.insert(key));
        }

        for idx in 0..val_dataset.len() {
            let (input, target) = val_dataset.get(idx).unwrap();
            let key = (input[0], input[1], target);
            assert!(seen.insert(key));
        }

        let total = ModularAdditionDataset::modulus() * ModularAdditionDataset::modulus();
        assert_eq!(seen.len(), total);
    }
}
