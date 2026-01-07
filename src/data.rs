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
/// where p = 97
#[derive(Debug, Clone)]
pub struct ModularAdditionDataset {
    /// All possible (a, b) pairs
    examples: Vec<(usize, usize, usize)>, // (a, b, target)
}

impl ModularAdditionDataset {
    const MODULUS: usize = 97;
    const VOCAB_SIZE: usize = 98; // 0-96 for numbers, 97 for '=' token
    const EQUALS_TOKEN: usize = 97;

    /// Create a new dataset with all 97x97 = 9409 possible pairs
    /// Split into train and validation sets (50/50) with fixed seed
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

        // Split 50/50
        let split_idx = all_examples.len() / 2;
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

        // Should split 9409 examples 50/50
        assert_eq!(train_dataset.len(), 4704);
        assert_eq!(val_dataset.len(), 4705);
        assert_eq!(train_dataset.len() + val_dataset.len(), 97 * 97);
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

            assert_eq!(equals, 97); // Equals token
            assert_eq!(target, (a + b) % 97);
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
}
