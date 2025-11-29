/* Warning: This file is manually written as C interface to rustymimi */

#ifndef RUSTYMIMI_H
#define RUSTYMIMI_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a Mimi tokenizer instance
typedef struct MimiTokenizer MimiTokenizer;

/// Create a new Mimi tokenizer from a safetensors file
///
/// @param path Path to the safetensors file (null-terminated C string)
/// @param num_codebooks Number of codebooks to use
/// @return Pointer to MimiTokenizer or NULL on error
MimiTokenizer* mimi_tokenizer_new(const char* path, size_t num_codebooks);

/// Encode PCM audio data to codes (streaming mode)
///
/// @param tokenizer Tokenizer instance
/// @param pcm_data PCM audio data (float32, shape: [1, 1, samples])
/// @param samples Number of samples
/// @param out_codes Output buffer for codes (will be allocated)
/// @param out_codebooks Output: number of codebooks
/// @param out_steps Output: number of time steps
/// @return 0 on success, -1 on error
int32_t mimi_encode_step(
    MimiTokenizer* tokenizer,
    const float* pcm_data,
    size_t samples,
    uint32_t** out_codes,
    size_t* out_codebooks,
    size_t* out_steps
);

/// Reset the tokenizer state
void mimi_reset(MimiTokenizer* tokenizer);

/// Free the tokenizer
void mimi_tokenizer_free(MimiTokenizer* tokenizer);

/// Free codes buffer allocated by mimi_encode_step
void mimi_free_codes(uint32_t* codes, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* RUSTYMIMI_H */
