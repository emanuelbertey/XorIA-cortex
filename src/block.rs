/*!
# xLSTM Block Implementation

This module implements the xLSTM block as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM block combines either sLSTM or mLSTM with layer normalization,
residual connections, and additional linear projections.
*/

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        PaddingConfig1d,
    },
    tensor::{activation, backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::{MLstm, MLstmconfig, MLstmstate, SLstm, SLstmconfig, SLstmstate, MinGru, MinGruConfig, MinGruState};

/// Type of LSTM block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    /// Scalar LSTM
    SLSTM,
    /// Matrix LSTM
    MLSTM,
    /// Minimal GRU
    MINGRU,
}

/// Configuration for xLSTM block
#[derive(Config, Debug)]
pub struct XLstmblockConfig {
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    #[config(default = "4")]
    pub num_heads: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Block type (sLSTM or mLSTM)
    pub block_type: BlockType,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:0.0}")]
    pub initializer: Initializer,
    /// Whether to use Causal Conv1D preprocessing
    #[config(default = "false")]
    pub use_conv: bool,
    /// Size of the Convolution kernel
    #[config(default = "4")]
    pub conv_kernel_size: usize,
    /// Whether to append an MLP mapping function
    #[config(default = "false")]
    pub use_mlp: bool,
}

impl XLstmblockConfig {
    /// Initialize a new xLSTM block
    pub fn init<B: Backend>(&self, device: &B::Device) -> XLstmblock<B> {
        let conv = if self.use_conv && self.conv_kernel_size > 0 {
            let pad = self.conv_kernel_size.saturating_sub(1);
            Some(
                Conv1dConfig::new(self.input_size, self.input_size, self.conv_kernel_size)
                    .with_padding(PaddingConfig1d::Explicit(pad))
                    .with_groups(self.input_size)
                    .init(device),
            )
        } else {
            None
        };

        let (mlp_fc1, mlp_fc2) = if self.use_mlp {
            let hidden_dim = self.input_size * 4;
            (
                Some(LinearConfig::new(self.input_size, hidden_dim).init(device)),
                Some(LinearConfig::new(hidden_dim, self.input_size).init(device)),
            )
        } else {
            (None, None)
        };

        let norm = LayerNormConfig::new(self.hidden_size).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        let proj = LinearConfig::new(self.hidden_size, self.input_size).init(device);

        match self.block_type {
            BlockType::SLSTM => {
                let lstm: SLstm<B> =
                    SLstmconfig::new(self.input_size, self.hidden_size, self.num_layers)
                        .with_dropout(self.dropout)
                        .with_initializer(self.initializer.clone())
                        .init(device);

                XLstmblock { lstm: LSTMVariant::SLSTM(lstm), norm, dropout, proj, conv, mlp_fc1, mlp_fc2 }
            }
            BlockType::MLSTM => {
                let lstm: MLstm<B> =
                    MLstmconfig::new(self.input_size, self.hidden_size, self.num_layers)
                        .with_num_heads(self.num_heads)
                        .with_dropout(self.dropout)
                        .with_initializer(self.initializer.clone())
                        .init(device);

                XLstmblock { lstm: LSTMVariant::MLSTM(lstm), norm, dropout, proj, conv, mlp_fc1, mlp_fc2 }
            }
            BlockType::MINGRU => {
                let gru: MinGru<B> =
                    MinGruConfig::new(self.input_size, self.hidden_size, self.num_layers)
                        .with_dropout(self.dropout)
                        .with_initializer(self.initializer.clone())
                        .init(device);

                XLstmblock { lstm: LSTMVariant::MINGRU(gru), norm, dropout, proj, conv, mlp_fc1, mlp_fc2 }
            }
        }
    }
}

/// Enum to hold either sLSTM or mLSTM
#[derive(Module, Debug)]
pub enum LSTMVariant<B: Backend> {
    /// Scalar LSTM variant
    SLSTM(SLstm<B>),
    /// Matrix LSTM variant
    MLSTM(MLstm<B>),
    /// Minimal GRU variant
    MINGRU(MinGru<B>),
}

/// Enum for holding either sLSTM or mLSTM states
#[derive(Debug, Clone)]
pub enum LSTMState<B: Backend> {
    /// States for sLSTM
    SLSTM(alloc::vec::Vec<SLstmstate<B, 2>>),
    /// States for mLSTM
    MLSTM(alloc::vec::Vec<MLstmstate<B>>),
    /// States for minGRU
    MINGRU(alloc::vec::Vec<MinGruState<B>>),
}

impl<B: Backend> LSTMState<B> {
    /// Detach the state from the computational graph
    pub fn detach(self) -> Self {
        match self {
            LSTMState::SLSTM(states) => {
                LSTMState::SLSTM(states.into_iter().map(|s| s.detach()).collect())
            }
            LSTMState::MLSTM(states) => {
                LSTMState::MLSTM(states.into_iter().map(|s| s.detach()).collect())
            }
            LSTMState::MINGRU(states) => {
                LSTMState::MINGRU(states.into_iter().map(|s| s.detach()).collect())
            }
        }
    }
}

/// xLSTM block combining LSTM with normalization and projections
#[derive(Module, Debug)]
pub struct XLstmblock<B: Backend> {
    /// LSTM variant (sLSTM or mLSTM)
    pub lstm: LSTMVariant<B>,
    /// Layer normalization
    pub norm: LayerNorm<B>,
    /// Dropout layer
    pub dropout: Dropout,
    /// Projection layer
    pub proj: Linear<B>,
    /// Optional Causal Conv1D block
    pub conv: Option<Conv1d<B>>,
    /// Optional MLP First Linear
    pub mlp_fc1: Option<Linear<B>>,
    /// Optional MLP Second Linear
    pub mlp_fc2: Option<Linear<B>>,
}

impl<B: Backend> XLstmblock<B> {
    /// Forward pass through xLSTM block
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor [`batch_size`, `seq_length`, `input_size`]
    /// * `state` - Optional initial state
    ///
    /// # Returns
    /// * Output tensor [`batch_size`, `seq_length`, `input_size`]
    /// * Final state
    pub fn forward(
        &self,
        input_seq: Tensor<B, 3>,
        state: Option<LSTMState<B>>,
    ) -> (Tensor<B, 3>, Option<LSTMState<B>>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive,
        B: Backend<FloatElem: num_traits::FromPrimitive>,
    {
        let mut conv_in = input_seq.clone();
        
        if let Some(conv) = &self.conv {
            let seq_len = conv_in.dims()[1];
            let mut x_conv = conv_in.swap_dims(1, 2);
            x_conv = conv.forward(x_conv);
            
            let b = x_conv.dims()[0];
            let c = x_conv.dims()[1];
            
            x_conv = x_conv.slice([0..b, 0..c, 0..seq_len]);
            x_conv = activation::gelu(x_conv);
            
            // Swap back
            conv_in = x_conv.swap_dims(1, 2);
        }

        let (mut out, new_state) = match (&self.lstm, state) {
            (LSTMVariant::SLSTM(lstm), Some(LSTMState::SLSTM(s))) => {
                let (o, ns) = lstm.forward(&conv_in, Some(s));
                (o, Some(LSTMState::SLSTM(ns)))
            }
            (LSTMVariant::SLSTM(lstm), _) => {
                let (o, ns) = lstm.forward(&conv_in, None);
                (o, Some(LSTMState::SLSTM(ns)))
            }
            (LSTMVariant::MLSTM(lstm), Some(LSTMState::MLSTM(s))) => {
                let (o, ns) = lstm.forward(&conv_in, Some(s));
                (o, Some(LSTMState::MLSTM(ns)))
            }
            (LSTMVariant::MLSTM(lstm), _) => {
                let (o, ns) = lstm.forward(&conv_in, None);
                (o, Some(LSTMState::MLSTM(ns)))
            }
            (LSTMVariant::MINGRU(gru), Some(LSTMState::MINGRU(s))) => {
                let (o, ns) = gru.forward(conv_in.clone(), Some(s));
                (o, Some(LSTMState::MINGRU(ns)))
            }
            (LSTMVariant::MINGRU(gru), _) => {
                let (o, ns) = gru.forward(conv_in.clone(), None);
                (o, Some(LSTMState::MINGRU(ns)))
            }
        };

        out = self.norm.forward(out);
        out = activation::gelu(out);
        out = self.proj.forward(out);
        
        if let (Some(fc1), Some(fc2)) = (&self.mlp_fc1, &self.mlp_fc2) {
            let mlp_hid = fc1.forward(out.clone());
            let mlp_hid = activation::gelu(mlp_hid);
            out = out + fc2.forward(mlp_hid);
        }
        
        out = self.dropout.forward(out);
        out = out + input_seq.clone();

        (out, new_state)
    }

    /// Get the block type
    pub const fn get_type(&self) -> BlockType {
        match &self.lstm {
            LSTMVariant::SLSTM(_) => BlockType::SLSTM,
            LSTMVariant::MLSTM(_) => BlockType::MLSTM,
            LSTMVariant::MINGRU(_) => BlockType::MINGRU,
        }
    }
}
