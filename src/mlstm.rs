/*
# mLSTM: Matrix Long Short-Term Memory (Canon Implementation)
Fiel a Beck et al. (2024). Sincronización perfecta entre modos Paralelo y Recurrente.
*/

use burn::{
    config::Config,
    module::Module,
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Clone, Debug)]
pub struct MLstmstate<B: Backend> {
    pub cell: Tensor<B, 4>,           // C_t: [B, H, D, D]
    pub hidden: Tensor<B, 2>,         // h_t: [B, D_hidden]
    pub normalizer: Tensor<B, 3>,     // n_t: [B, H, D]
    pub max_gate_log: Tensor<B, 3>,   // m_t: [B, H, 1]
}

impl<B: Backend> MLstmstate<B> {
    pub const fn new(cell: Tensor<B, 4>, hidden: Tensor<B, 2>, normalizer: Tensor<B, 3>, max_gate_log: Tensor<B, 3>) -> Self {
        Self { cell, hidden, normalizer, max_gate_log }
    }
    pub fn detach(self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
        }
    }
}

#[derive(Config, Debug)]
pub struct MLstmconfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    #[config(default = "4")]
    pub num_heads: usize,
    #[config(default = "2.0")]
    pub proj_factor: f32,
    #[config(default = "Initializer::Normal{mean: 0.0, std: 0.02}")]
    pub initializer: Initializer,
    #[config(default = "0.0")]
    pub forget_bias: f32,
    #[config(default = "-1.0")] 
    pub input_bias: f32,
    #[config(default = "0.0")]
    pub dropout: f64,
}

impl MLstmconfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLstm<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                MLstmcell::new(input_size, self.d_hidden, self.num_heads, self, device)
            })
            .collect();
        MLstm { layers, d_hidden: self.d_hidden, num_layers: self.num_layers }
    }
}

#[derive(Module, Debug)]
pub struct MLstm<B: Backend> {
    pub layers: alloc::vec::Vec<MLstmcell<B>>,
    pub d_hidden: usize,
    pub num_layers: usize,
}

impl<B: Backend> MLstm<B> {
    pub fn forward(&self, input_seq: &Tensor<B, 3>, states: Option<alloc::vec::Vec<MLstmstate<B>>>) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>) {
        self.forward_ext(input_seq, states, false)
    }

    pub fn forward_ext(&self, input_seq: &Tensor<B, 3>, states: Option<alloc::vec::Vec<MLstmstate<B>>>, frozen: bool) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>) {
        let device = input_seq.device();
        let [batch_size, _, _] = input_seq.dims();
        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));
        let mut layer_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (h_seq, new_state) = layer.forward_sequence_ext(&layer_input, hidden_states[layer_idx].clone(), frozen);
            hidden_states[layer_idx] = new_state;
            layer_input = h_seq;
        }
        (layer_input, hidden_states)
    }

    fn init_hidden(&self, batch_size: usize, device: &B::Device) -> alloc::vec::Vec<MLstmstate<B>> {
        let internal_size = (self.d_hidden as f32 * self.layers[0].proj_factor) as usize;
        let head_dim = internal_size / self.layers[0].num_heads;
        (0..self.num_layers).map(|_| {
            MLstmstate::new(
                Tensor::zeros([batch_size, self.layers[0].num_heads, head_dim, head_dim], device),
                Tensor::zeros([batch_size, self.d_hidden], device),
                Tensor::zeros([batch_size, self.layers[0].num_heads, head_dim], device),
                Tensor::zeros([batch_size, self.layers[0].num_heads, 1], device).add_scalar(-10.0),
            )
        }).collect()
    }
}

#[derive(Module, Debug)]
pub struct MLstmcell<B: Backend> {
    pub w_q: Linear<B>,
    pub w_k: Linear<B>,
    pub w_v: Linear<B>,
    pub w_gates: Linear<B>,
    pub w_proj: Linear<B>,
    pub ln: LayerNorm<B>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub attention_scale: f32,
    pub proj_factor: f32,
}

impl<B: Backend> MLstmcell<B> {
    pub fn new(input_size: usize, hidden_size: usize, num_heads: usize, config: &MLstmconfig, device: &B::Device) -> Self {
        let internal_size = (hidden_size as f32 * config.proj_factor) as usize;
        let head_dim = internal_size / num_heads;
        
        Self {
            w_q: LinearConfig::new(input_size, internal_size).with_bias(false).init(device),
            w_k: LinearConfig::new(input_size, internal_size).with_bias(false).init(device),
            w_v: LinearConfig::new(input_size, internal_size).with_bias(false).init(device),
            w_gates: LinearConfig::new(input_size, 3 * num_heads).with_bias(true).init(device),
            w_proj: LinearConfig::new(internal_size, hidden_size).with_bias(true).init(device),
            ln: LayerNormConfig::new(head_dim).init(device),
            num_heads,
            head_dim,
            attention_scale: 1.0 / (head_dim as f32).sqrt(),
            proj_factor: config.proj_factor,
        }
    }

    pub fn forward_sequence_ext(&self, input_seq: &Tensor<B, 3>, state: MLstmstate<B>, frozen: bool) -> (Tensor<B, 3>, MLstmstate<B>) {
        let [batch_size, seq_len, _] = input_seq.dims();
        let device = input_seq.device();

        // 1. Projections
        let q = (self.w_q.forward(input_seq.clone()) * self.attention_scale).reshape([batch_size, seq_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let k = self.w_k.forward(input_seq.clone()).reshape([batch_size, seq_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        let v = self.w_v.forward(input_seq.clone()).reshape([batch_size, seq_len, self.num_heads, self.head_dim]).swap_dims(1, 2);
        
        let gates = self.w_gates.forward(input_seq.clone());
        let i_log = gates.clone().slice([0..batch_size, 0..seq_len, 0..self.num_heads]).swap_dims(1, 2).reshape([batch_size, self.num_heads, seq_len, 1]);
        let f_log = gates.clone().slice([0..batch_size, 0..seq_len, self.num_heads..2*self.num_heads]).swap_dims(1, 2).reshape([batch_size, self.num_heads, seq_len, 1]);
        let o_log = gates.slice([0..batch_size, 0..seq_len, 2*self.num_heads..3*self.num_heads]).swap_dims(1, 2).reshape([batch_size, self.num_heads, seq_len, 1]);

        // 2. Stable Parallel Decay
        let tril = Tensor::<B, 2>::tril(Tensor::ones([seq_len, seq_len], &device), 0).reshape([1, 1, seq_len, seq_len]);
        let f_log_cumsum = tril.clone().matmul(f_log.clone());
        
        let log_l = (f_log_cumsum.clone() - f_log_cumsum.clone().swap_dims(2, 3)) + i_log.clone().swap_dims(2, 3);
        let log_l_masked = log_l.mask_fill(tril.equal(Tensor::zeros([1, 1, seq_len, seq_len], &device)), -1e30);
        
        let m_0 = state.max_gate_log.clone().reshape([batch_size, self.num_heads, 1, 1]);
        let log_init = f_log_cumsum.clone() + m_0;
        let m_t = log_l_masked.clone().max_dim(3).max_pair(log_init.clone());
        
        let weights = (log_l_masked - m_t.clone()).exp();
        let init_scale = (log_init - m_t.clone()).exp();

        // 3. Associative Read (Eq 21)
        let num_parallel = (weights.clone() * q.clone().matmul(k.clone().swap_dims(2, 3))).matmul(v.clone());
        let h_initial_raw = q.clone().matmul(state.cell.clone().swap_dims(2, 3)) * init_scale.clone();
        let h_tilde_raw = num_parallel + h_initial_raw; 

        let den_parallel = weights.clone().matmul(k.clone());
        let den_init = init_scale.clone() * state.normalizer.clone().reshape([batch_size, self.num_heads, 1, self.head_dim]);
        let den_vec = den_parallel + den_init;
        let den = (q.clone() * den_vec).sum_dim(3).abs().reshape([batch_size, self.num_heads, seq_len, 1]).max_pair(Tensor::ones_like(&m_t));

        // SYNC: LN(raw) -> Div -> Gate
        let h_tilde_norm = self.ln.forward(h_tilde_raw);
        let h_t = (h_tilde_norm / den) * (o_log - m_t.clone()).exp();
        
        let h_combined = h_t.swap_dims(1, 2).reshape([batch_size, seq_len, self.num_heads * self.head_dim]);
        let y_seq = self.w_proj.forward(h_combined);

        // 4. State Update
        let last = seq_len - 1;
        let final_state = if frozen { state } else {
            let h_last = y_seq.clone().slice([0..batch_size, last..last+1]).reshape([batch_size, y_seq.dims()[2]]);
            let last_w = weights.slice([0..batch_size, 0..self.num_heads, last..last+1, 0..seq_len]);
            let last_s = init_scale.clone().slice([0..batch_size, 0..self.num_heads, last..last+1, 0..1]);

            MLstmstate::new(
                (state.cell * last_s.clone().reshape([batch_size, self.num_heads, 1, 1])) + (v * last_w.clone().reshape([batch_size, self.num_heads, seq_len, 1])).swap_dims(2, 3).matmul(k.clone()),
                h_last,
                (state.normalizer * last_s.reshape([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, self.head_dim])) + last_w.matmul(k).reshape([batch_size, self.num_heads, self.head_dim]),
                m_t.slice([0..batch_size, 0..self.num_heads, last..last+1, 0..1]).reshape([batch_size, self.num_heads, 1]),
            )
        };
        (y_seq, final_state)
    }

    pub fn forward_sequence(&self, input_seq: &Tensor<B, 3>, state: MLstmstate<B>) -> (Tensor<B, 3>, MLstmstate<B>) {
        self.forward_sequence_ext(input_seq, state, false)
    }

    pub fn forward_step(&self, input: &Tensor<B, 2>, state: MLstmstate<B>, frozen: bool) -> (Tensor<B, 2>, MLstmstate<B>) {
        let [batch_size, _] = input.dims();
        let gates = self.w_gates.forward(input.clone());
        let chunk = gates.chunk(3, 1);
        
        // UNIFIED 4D SHAPES
        let i_log = chunk[0].clone().reshape([batch_size, self.num_heads, 1, 1]);
        let f_log = chunk[1].clone().reshape([batch_size, self.num_heads, 1, 1]);
        let o_log = chunk[2].clone().reshape([batch_size, self.num_heads, 1, 1]);

        let q = (self.w_q.forward(input.clone()) * self.attention_scale).reshape([batch_size, self.num_heads, 1, self.head_dim]);
        let k = self.w_k.forward(input.clone()).reshape([batch_size, self.num_heads, 1, self.head_dim]);
        let v = self.w_v.forward(input.clone()).reshape([batch_size, self.num_heads, 1, self.head_dim]);

        let m_prev = state.max_gate_log.clone().reshape([batch_size, self.num_heads, 1, 1]);
        let m_t = f_log.clone().add(m_prev.clone()).max_pair(i_log.clone());
        let f_hat = (f_log + m_prev - m_t.clone()).exp();
        let i_hat = (i_log - m_t.clone()).exp();

        // Update Cell & Norm
        let cell = state.cell.clone() * f_hat.clone() + v.clone().swap_dims(2, 3).matmul(k.clone()) * i_hat.clone();
        let norm = state.normalizer.clone().reshape([batch_size, self.num_heads, 1, self.head_dim]) * f_hat.clone() + k.clone() * i_hat.clone();

        // Read (Q @ C^T) in 4D
        let h_tilde_raw = q.clone().matmul(cell.clone().swap_dims(2, 3));
        let den = (q.clone() * norm.clone()).sum_dim(3).abs().reshape([batch_size, self.num_heads, 1, 1]).max_pair(Tensor::ones_like(&m_t));

        // SYNC FIX: LN(raw) -> Div -> Gate
        let h_tilde_norm = self.ln.forward(h_tilde_raw);
        let h_t = (h_tilde_norm / den) * (o_log - m_t.clone()).exp();
        
        let y = self.w_proj.forward(h_t.reshape([batch_size, self.num_heads * self.head_dim]));

        let new_state = if frozen { state } else { 
            MLstmstate::new(
                cell, 
                y.clone(), 
                norm.reshape([batch_size, self.num_heads, self.head_dim]), 
                m_t.reshape([batch_size, self.num_heads, 1])
            ) 
        };

        (y, new_state)
    }

    pub fn forward(&self, input: &Tensor<B, 2>, state: MLstmstate<B>) -> (Tensor<B, 2>, MLstmstate<B>) {
        self.forward_step(input, state, false)
    }
}