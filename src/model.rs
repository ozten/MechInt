use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

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
            n_layers: 2,          // transformer layers
        }
    }
}

/// Decoder-only transformer for modular addition
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    layers: Vec<TransformerLayer<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
    d_model: usize,
    seq_length: usize,
}

#[derive(Module, Debug)]
struct TransformerLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    ln1: LayerNorm<B>,
    mlp: MLP<B>,
    ln2: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Gelu,
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

        // Final layer norm
        let ln_f = LayerNormConfig::new(self.d_model).init(device);

        // Language model head (d_model -> vocab_size)
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .init(device);

        Transformer {
            token_embedding,
            position_embedding,
            layers,
            ln_f,
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

        // Apply transformer layers
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }

        // Final layer norm
        hidden = self.ln_f.forward(hidden);

        // Take the last position (after '=') for prediction
        let last_hidden = hidden.slice([0..batch_size, (seq_length - 1)..seq_length])
            .squeeze();

        // Project to vocabulary
        self.lm_head.forward(last_hidden)
    }
}

impl<B: Backend> TransformerLayer<B> {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, device: &B::Device) -> Self {
        let attention = MultiHeadAttentionConfig::new(d_model, n_heads)
            .init(device);

        let ln1 = LayerNormConfig::new(d_model).init(device);
        let ln2 = LayerNormConfig::new(d_model).init(device);

        let mlp = MLP::new(d_model, d_ff, device);

        Self {
            attention,
            ln1,
            mlp,
            ln2,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm architecture
        // Self-attention with residual
        // Note: Causal masking not strictly needed for this task since we see full input [a,b,=]
        // before predicting. Standard attention is sufficient.
        let normed = self.ln1.forward(x.clone());
        let mha_input = MhaInput::self_attn(normed);
        let mha_output = self.attention.forward(mha_input);
        let x = x + mha_output.context;

        // MLP with residual
        let normed = self.ln2.forward(x.clone());
        let mlp_out = self.mlp.forward(normed);
        x + mlp_out
    }
}

impl<B: Backend> MLP<B> {
    fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(d_model, d_ff).init(device);
        let fc2 = LinearConfig::new(d_ff, d_model).init(device);
        let activation = Gelu::new();

        Self { fc1, fc2, activation }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

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
        let seq_length = 3;

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
}
