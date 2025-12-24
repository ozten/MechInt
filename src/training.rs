use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{data::ModularAdditionBatch, model::Transformer};

impl<B: Backend> Transformer<B> {
    /// Forward pass with loss for classification metrics.
    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<ModularAdditionBatch<B>, ClassificationOutput<B>>
    for Transformer<B>
{
    fn step(&self, batch: ModularAdditionBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ModularAdditionBatch<B>, ClassificationOutput<B>> for Transformer<B> {
    fn step(&self, batch: ModularAdditionBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}
