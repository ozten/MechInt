use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{
            generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig,
        },
        Embedding, EmbeddingConfig, Linear, LinearConfig, Relu,
    },
    tensor::{backend::Backend, Bool, Int, Tensor},
};
use burn::tensor::module::embedding;

use crate::data::ModularAdditionDataset;

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub seq_length: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: ModularAdditionDataset::vocab_size(),
            seq_length: ModularAdditionDataset::sequence_length(),
            d_model: 128,         // embedding dimension
            n_heads: 4,           // attention heads
            d_ff: 512,            // MLP hidden dimension
            n_layers: 1,          // transformer layers
        }
    }
}

/// Decoder-only transformer for modular addition
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    layers: Vec<TransformerLayer<B>>,
    lm_head: Linear<B>,
    d_model: usize,
    seq_length: usize,
}

#[derive(Module, Debug)]
struct TransformerLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    mlp: MLP<B>,
}

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
}

impl TransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        // Token embeddings (vocab -> d_model)
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model)
            .init(device);

        // Learned positional embeddings
        let position_embedding = EmbeddingConfig::new(self.seq_length, self.d_model)
            .init(device);

        // Create transformer layers
        let mut layers = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            layers.push(TransformerLayer::new(
                self.d_model,
                self.n_heads,
                self.d_ff,
                device,
            ));
        }

        // Language model head (d_model -> vocab_size)
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .init(device);

        Transformer {
            token_embedding,
            position_embedding,
            layers,
            lm_head,
            d_model: self.d_model,
            seq_length: self.seq_length,
        }
    }
}

impl<B: Backend> Transformer<B> {
    /// Get the token embedding for a single token (for analysis)
    /// Returns shape [1, embedding_dim]
    pub fn get_token_embedding(&self, token_id: usize) -> Tensor<B, 2> {
        let device = self.token_embedding.devices()[0].clone();
        let token_vec = vec![token_id as i32];
        let input = Tensor::<B, 1, Int>::from_ints(token_vec.as_slice(), &device)
            .reshape([1, 1]);
        let embedding = self.token_embedding.forward(input); // [1, 1, embedding_dim]
        embedding.squeeze::<2>() // Remove seq dimension -> [1, embedding_dim]
    }

    pub fn token_embedding_weights(&self) -> Tensor<B, 2> {
        self.token_embedding.weight.val()
    }

    /// Forward pass
    /// Input: [batch_size, seq_length] of token indices
    /// Output: [batch_size, vocab_size] logits for next token prediction
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch_size, seq_length] = x.dims();
        let device = x.device();

        // Token embeddings: [batch_size, seq_length, d_model]
        let tok_emb = self.token_embedding.forward(x);

        // Position embeddings: [batch_size, seq_length, d_model]
        let positions = Tensor::arange(0..seq_length as i64, &device)
            .reshape([1, seq_length])
            .repeat(&[batch_size, 1]);
        let pos_emb = self.position_embedding.forward(positions);

        // Add token and position embeddings
        let mut hidden = tok_emb + pos_emb;

        let attn_mask = generate_autoregressive_mask::<B>(batch_size, seq_length, &device);

        // Apply transformer layers
        for layer in &self.layers {
            hidden = layer.forward(hidden, &attn_mask);
        }

        // Take the last position (after '=') for prediction
        let last_hidden = hidden.slice([0..batch_size, (seq_length - 1)..seq_length])
            .squeeze();

        // Project to vocabulary
        self.lm_head.forward(last_hidden)
    }

    pub fn forward_with_token_weights(
        &self,
        x: Tensor<B, 2, Int>,
        token_weights: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch_size, seq_length] = x.dims();
        let device = x.device();

        // Token embeddings: [batch_size, seq_length, d_model]
        let tok_emb = embedding(token_weights, x);

        // Position embeddings: [batch_size, seq_length, d_model]
        let positions = Tensor::arange(0..seq_length as i64, &device)
            .reshape([1, seq_length])
            .repeat(&[batch_size, 1]);
        let pos_emb = self.position_embedding.forward(positions);

        // Add token and position embeddings
        let mut hidden = tok_emb + pos_emb;

        let attn_mask = generate_autoregressive_mask::<B>(batch_size, seq_length, &device);

        // Apply transformer layers
        for layer in &self.layers {
            hidden = layer.forward(hidden, &attn_mask);
        }

        // Take the last position (after '=') for prediction
        let last_hidden = hidden
            .slice([0..batch_size, (seq_length - 1)..seq_length])
            .squeeze();

        // Project to vocabulary
        self.lm_head.forward(last_hidden)
    }

    /// Forward pass that also returns post-ReLU MLP activations for the final token.
    pub fn forward_with_mlp_activations(
        &self,
        x: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch_size, seq_length] = x.dims();
        let device = x.device();

        let tok_emb = self.token_embedding.forward(x);

        let positions = Tensor::arange(0..seq_length as i64, &device)
            .reshape([1, seq_length])
            .repeat(&[batch_size, 1]);
        let pos_emb = self.position_embedding.forward(positions);

        let mut hidden = tok_emb + pos_emb;

        let attn_mask = generate_autoregressive_mask::<B>(batch_size, seq_length, &device);

        let mut last_mlp_acts: Option<Tensor<B, 3>> = None;
        for layer in &self.layers {
            let (next_hidden, mlp_acts) = layer.forward_with_mlp_activations(hidden, &attn_mask);
            hidden = next_hidden;
            last_mlp_acts = Some(mlp_acts);
        }

        let mlp_acts = last_mlp_acts.expect("transformer must have at least one layer");
        let last_mlp = mlp_acts
            .slice([0..batch_size, (seq_length - 1)..seq_length])
            .squeeze();

        let last_hidden = hidden
            .slice([0..batch_size, (seq_length - 1)..seq_length])
            .squeeze();

        let logits = self.lm_head.forward(last_hidden);
        (logits, last_mlp)
    }
}

impl<B: Backend> TransformerLayer<B> {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .init(device);

        let mlp = MLP::new(d_model, d_ff, device);

        Self { attention, mlp }
    }

    fn forward(&self, x: Tensor<B, 3>, attn_mask: &Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        // Self-attention with causal mask and residual connection.
        let mha_input = MhaInput::self_attn(x.clone()).mask_attn(attn_mask.clone());
        let mha_output = self.attention.forward(mha_input);
        let x = x + mha_output.context;

        // MLP with residual
        let mlp_out = self.mlp.forward(x.clone());
        x + mlp_out
    }

    fn forward_with_mlp_activations(
        &self,
        x: Tensor<B, 3>,
        attn_mask: &Tensor<B, 3, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mha_input = MhaInput::self_attn(x.clone()).mask_attn(attn_mask.clone());
        let mha_output = self.attention.forward(mha_input);
        let x = x + mha_output.context;

        let (mlp_out, mlp_acts) = self.mlp.forward_with_activations(x.clone());
        (x + mlp_out, mlp_acts)
    }
}

impl<B: Backend> MLP<B> {
    fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(d_model, d_ff).init(device);
        let fc2 = LinearConfig::new(d_ff, d_model).init(device);
        let activation = Relu::new();

        Self { fc1, fc2, activation }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        self.fc2.forward(x)
    }

    fn forward_with_activations(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = self.fc1.forward(x);
        let acts = self.activation.forward(x);
        let out = self.fc2.forward(acts.clone());
        (out, acts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use std::any::type_name;

    type TestBackend = NdArray;

    #[test]
    fn test_model_creation() {
        let device = Default::default();
        let config = TransformerConfig::default();
        let _model = config.init::<TestBackend>(&device);
    }

    #[test]
    fn test_forward_pass() {
        let device = Default::default();
        let config = TransformerConfig::default();
        let model = config.init::<TestBackend>(&device);

        // Create a batch of input sequences
        let batch_size = 4;
        let _seq_length = 3;

        // Dummy input: [[0, 1, 113], [2, 3, 113], ...]
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            [[0, 1, 113], [2, 3, 113], [5, 10, 113], [20, 30, 113]],
            &device,
        );

        // Forward pass
        let logits = model.forward(input);

        // Check output shape: [batch_size, vocab_size]
        assert_eq!(logits.dims(), [batch_size, ModularAdditionDataset::vocab_size()]);
    }

    #[test]
    fn test_config_matches_spec() {
        let config = TransformerConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.d_ff, 512);
        assert_eq!(config.n_layers, 1);
        assert_eq!(config.seq_length, 3);
        assert_eq!(config.d_model / config.n_heads, 32);
    }

    #[test]
    fn test_embeddings_are_learned_and_sized() {
        let device = Default::default();
        let config = TransformerConfig::default();
        let model = config.init::<TestBackend>(&device);

        let [vocab_size, d_model] = model.token_embedding.weight.dims();
        assert_eq!(vocab_size, ModularAdditionDataset::vocab_size());
        assert_eq!(d_model, config.d_model);

        let [pos_length, pos_dim] = model.position_embedding.weight.dims();
        assert_eq!(pos_length, config.seq_length);
        assert_eq!(pos_dim, config.d_model);
    }

    #[test]
    fn test_biases_enabled() {
        let device = Default::default();
        let model = TransformerConfig::default().init::<TestBackend>(&device);
        let layer = &model.layers[0];

        assert!(model.lm_head.bias.is_some());
        assert!(layer.attention.query.bias.is_some());
        assert!(layer.attention.key.bias.is_some());
        assert!(layer.attention.value.bias.is_some());
        assert!(layer.attention.output.bias.is_some());
        assert!(layer.mlp.fc1.bias.is_some());
        assert!(layer.mlp.fc2.bias.is_some());
    }

    #[test]
    fn test_activation_is_relu() {
        let device = Default::default();
        let model = TransformerConfig::default().init::<TestBackend>(&device);
        let activation_name = std::any::type_name_of_val(&model.layers[0].mlp.activation);
        assert_eq!(activation_name, type_name::<Relu>());
    }

    #[test]
    fn test_no_layer_norm_modules() {
        let device = Default::default();
        let model = TransformerConfig::default().init::<TestBackend>(&device);
        let display = format!("{model}");
        assert!(!display.contains("LayerNorm"));
    }

    #[test]
    fn test_causal_mask_applied_in_layer() {
        let device = Default::default();
        let layer = TransformerLayer::new(8, 2, 16, &device);
        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new((0..24).map(|v| v as f32).collect(), [1, 3, 8]),
            &device,
        );
        let mask = generate_autoregressive_mask::<TestBackend>(1, 3, &device);

        let expected_masked = {
            let mha_input = MhaInput::self_attn(input.clone()).mask_attn(mask.clone());
            let mha_output = layer.attention.forward(mha_input);
            let x = input.clone() + mha_output.context;
            let mlp_out = layer.mlp.forward(x.clone());
            x + mlp_out
        };

        let expected_unmasked = {
            let mha_input = MhaInput::self_attn(input.clone());
            let mha_output = layer.attention.forward(mha_input);
            let x = input.clone() + mha_output.context;
            let mlp_out = layer.mlp.forward(x.clone());
            x + mlp_out
        };

        let actual = layer.forward(input, &mask);

        let actual_vec = actual.into_data().to_vec::<f32>().unwrap();
        let masked_vec = expected_masked.into_data().to_vec::<f32>().unwrap();
        let unmasked_vec = expected_unmasked.into_data().to_vec::<f32>().unwrap();

        assert_eq!(actual_vec, masked_vec);
        assert_ne!(masked_vec, unmasked_vec);
    }
}
