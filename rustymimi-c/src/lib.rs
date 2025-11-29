// Copyright (c) MoshiMac - C FFI wrapper for rustymimi
// Provides a C interface to the Rust Mimi implementation for use in Swift

use moshi::{candle, candle_nn, mimi, seanet, transformer, conv, NormType};
use std::ffi::{c_char, CStr};
use std::ptr;
use std::slice;

/// Opaque handle to a Mimi tokenizer instance
#[repr(C)]
pub struct MimiTokenizer {
    _private: [u8; 0],
}

fn mimi_cfg(num_codebooks: usize) -> mimi::Config {
    let seanet_cfg = seanet::Config {
        dimension: 512,
        channels: 1,
        causal: true,
        n_filters: 64,
        n_residual_layers: 1,
        activation: candle_nn::Activation::Elu(1.),
        compress: 2,
        dilation_base: 2,
        disable_norm_outer_blocks: 0,
        final_activation: None,
        kernel_size: 7,
        residual_kernel_size: 3,
        last_kernel_size: 3,
        lstm: 0,
        norm: conv::Norm::WeightNorm,
        pad_mode: conv::PadMode::Constant,
        ratios: vec![8, 6, 5, 4],
        true_skip: true,
    };
    let transformer_cfg = transformer::Config {
        d_model: seanet_cfg.dimension,
        num_heads: 8,
        num_layers: 8,
        causal: true,
        norm_first: true,
        bias_ff: false,
        bias_attn: false,
        layer_scale: Some(0.01),
        context: 250,
        conv_kernel_size: 5,
        use_conv_bias: true,
        use_conv_block: false,
        max_period: 10000,
        positional_embedding: transformer::PositionalEmbedding::Rope,
        gating: None,
        norm: NormType::LayerNorm,
        dim_feedforward: 2048,
        kv_repeat: 1,
        conv_layout: true,
        cross_attention: None,
        shared_cross_attn: true,
        max_seq_len: 8192,
    };
    mimi::Config {
        channels: 1,
        sample_rate: 24_000.,
        frame_rate: 12.5,
        renormalize: true,
        resample_method: mimi::ResampleMethod::Conv,
        seanet: seanet_cfg,
        transformer: transformer_cfg,
        quantizer_n_q: num_codebooks,
        quantizer_bins: 2048,
        quantizer_dim: 256,
    }
}

struct MimiTokenizerImpl {
    mimi: mimi::Mimi,
    device: candle::Device,
    dtype: candle::DType,
}

/// Create a new Mimi tokenizer from a safetensors file
///
/// # Arguments
/// * `path` - Path to the safetensors file (null-terminated C string)
/// * `num_codebooks` - Number of codebooks to use
///
/// # Returns
/// Pointer to MimiTokenizer or null on error
#[no_mangle]
pub extern "C" fn mimi_tokenizer_new(
    path: *const c_char,
    num_codebooks: usize,
) -> *mut MimiTokenizer {
    if path.is_null() {
        return ptr::null_mut();
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let result = (|| -> Result<_, anyhow::Error> {
        let device = candle::Device::Cpu;
        let dtype = candle::DType::F32;
        let path_buf = std::path::PathBuf::from(path_str);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path_buf], dtype, &device)?
        };

        let cfg = mimi_cfg(num_codebooks);
        let mimi = mimi::Mimi::new(cfg, vb)?;

        let tokenizer = Box::new(MimiTokenizerImpl { mimi, device, dtype });
        Ok(Box::into_raw(tokenizer) as *mut MimiTokenizer)
    })();

    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            eprintln!("Error creating Mimi tokenizer: {}", e);
            ptr::null_mut()
        }
    }
}

/// Encode PCM audio data to codes (streaming mode)
///
/// # Arguments
/// * `tokenizer` - Tokenizer instance
/// * `pcm_data` - PCM audio data (float32, shape: [1, 1, samples])
/// * `samples` - Number of samples
/// * `out_codes` - Output buffer for codes (will be allocated)
/// * `out_codebooks` - Output: number of codebooks
/// * `out_steps` - Output: number of time steps
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn mimi_encode_step(
    tokenizer: *mut MimiTokenizer,
    pcm_data: *const f32,
    samples: usize,
    out_codes: *mut *mut u32,
    out_codebooks: *mut usize,
    out_steps: *mut usize,
) -> i32 {
    if tokenizer.is_null() || pcm_data.is_null() || out_codes.is_null() {
        return -1;
    }

    let tokenizer = unsafe { &mut *(tokenizer as *mut MimiTokenizerImpl) };
    let pcm_slice = unsafe { slice::from_raw_parts(pcm_data, samples) };

    let result = (|| -> Result<_, anyhow::Error> {
        let pcm_tensor = candle::Tensor::from_slice(pcm_slice, (1, 1, samples), &tokenizer.device)?
            .to_dtype(tokenizer.dtype)?;

        let codes = tokenizer.mimi.encode_step(&pcm_tensor.into(), &().into())?;

        match codes.as_option() {
            Some(codes_tensor) => {
                let codes_vec = codes_tensor.to_vec3::<u32>()?;
                if codes_vec.is_empty() || codes_vec[0].is_empty() {
                    return Err(anyhow::anyhow!("Empty codes returned"));
                }

                let codebooks = codes_vec[0].len();
                let steps = codes_vec[0][0].len();

                // Flatten codes: [codebooks, steps] -> [codebooks * steps]
                let mut flat_codes = Vec::with_capacity(codebooks * steps);
                for step_idx in 0..steps {
                    for cb_idx in 0..codebooks {
                        flat_codes.push(codes_vec[0][cb_idx][step_idx]);
                    }
                }

                let boxed_codes = flat_codes.into_boxed_slice();
                let codes_ptr = Box::into_raw(boxed_codes) as *mut u32;

                unsafe {
                    *out_codes = codes_ptr;
                    *out_codebooks = codebooks;
                    *out_steps = steps;
                }

                Ok(())
            }
            None => Err(anyhow::anyhow!("No codes returned")),
        }
    })();

    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error encoding: {}", e);
            -1
        }
    }
}

/// Reset the tokenizer state
#[no_mangle]
pub extern "C" fn mimi_reset(tokenizer: *mut MimiTokenizer) {
    if tokenizer.is_null() {
        return;
    }
    let tokenizer = unsafe { &mut *(tokenizer as *mut MimiTokenizerImpl) };
    tokenizer.mimi.reset_state();
}

/// Free the tokenizer
#[no_mangle]
pub extern "C" fn mimi_tokenizer_free(tokenizer: *mut MimiTokenizer) {
    if !tokenizer.is_null() {
        unsafe {
            let _ = Box::from_raw(tokenizer as *mut MimiTokenizerImpl);
        }
    }
}

/// Free codes buffer allocated by mimi_encode_step
#[no_mangle]
pub extern "C" fn mimi_free_codes(codes: *mut u32, size: usize) {
    if !codes.is_null() {
        unsafe {
            let _ = Box::from_raw(slice::from_raw_parts_mut(codes, size));
        }
    }
}
