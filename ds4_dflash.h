#ifndef DS4_DFLASH_H
#define DS4_DFLASH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_DFLASH_MAX_TARGET_LAYERS 128u
#define DS4_DFLASH_MAX_PATH 4096u
#define DS4_DFLASH_MAX_TENSOR_NAME 160u

typedef enum {
    DS4_DFLASH_TENSOR_UNKNOWN = 0,
    DS4_DFLASH_TENSOR_BF16,
    DS4_DFLASH_TENSOR_BOOL,
    DS4_DFLASH_TENSOR_I64,
} ds4_dflash_tensor_dtype;

typedef struct {
    char name[DS4_DFLASH_MAX_TENSOR_NAME];
    ds4_dflash_tensor_dtype dtype;
    uint32_t ndim;
    uint64_t shape[4];
    uint64_t data_offsets[2];
    uint64_t abs_offset;
    uint64_t nbytes;
} ds4_dflash_tensor;

typedef struct {
    char source_path[DS4_DFLASH_MAX_PATH];
    uint32_t block_size;
    uint32_t mask_token_id;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t draft_vocab_size;
    uint32_t target_hidden_size;
    uint32_t num_target_layers;
    uint32_t num_hidden_layers;
    uint32_t intermediate_size;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t hc_mult;
    uint32_t sliding_window;
    uint32_t max_anchors;
    float rope_theta;
    bool sliding_window_non_causal;
    uint32_t target_layer_ids[DS4_DFLASH_MAX_TARGET_LAYERS];
    uint32_t n_target_layer_ids;
    bool loaded;
} ds4_dflash_config;

typedef struct {
    char source_path[DS4_DFLASH_MAX_PATH];
    uint64_t header_len;
    uint64_t file_size;
    uint64_t data_start;
    void *map;
    ds4_dflash_tensor *tensors;
    uint32_t n_tensors;
    uint32_t n_bound_tensors;
    int fd;
    bool loaded;
} ds4_dflash_weights;

typedef struct {
    float *hidden;
    uint32_t *positions;
    uint32_t capacity;
    uint32_t len;
    uint32_t start;
    uint32_t hidden_size;
} ds4_dflash_hidden_history;

typedef struct {
    uint32_t drafted;
    uint32_t verified;
    uint32_t accepted_including_anchor;
    uint32_t misses;
    uint32_t rejected_draft_tokens;
    int miss_index;
    int miss_draft_token;
    int miss_target_token;
    int miss_target_top;
} ds4_dflash_verify_stats;

void ds4_dflash_config_init(ds4_dflash_config *cfg);
void ds4_dflash_config_free(ds4_dflash_config *cfg);
void ds4_dflash_weights_init(ds4_dflash_weights *w);
void ds4_dflash_weights_free(ds4_dflash_weights *w);
void ds4_dflash_hidden_history_init(ds4_dflash_hidden_history *h);
void ds4_dflash_hidden_history_free(ds4_dflash_hidden_history *h);
void ds4_dflash_hidden_history_reset(ds4_dflash_hidden_history *h);
void ds4_dflash_hidden_history_rewind(ds4_dflash_hidden_history *h,
                                      uint32_t position_exclusive);

void ds4_dflash_verify_stats_init(ds4_dflash_verify_stats *stats,
                                  uint32_t drafted,
                                  uint32_t accepted_including_anchor);

bool ds4_dflash_verify_step(ds4_dflash_verify_stats *stats,
                            uint32_t index,
                            int draft_token,
                            int target_token,
                            int target_top);

int ds4_dflash_hidden_history_reserve(ds4_dflash_hidden_history *h,
                                      const ds4_dflash_config *cfg,
                                      uint32_t capacity,
                                      char *err,
                                      size_t errlen);

int ds4_dflash_hidden_history_append(ds4_dflash_hidden_history *h,
                                     uint32_t position,
                                     const float *hidden,
                                     char *err,
                                     size_t errlen);

uint32_t ds4_dflash_hidden_history_count_visible(const ds4_dflash_hidden_history *h,
                                                 uint32_t anchor_position,
                                                 uint32_t max_rows);

/* Copies visible rows before anchor_position in chronological order. max_rows
 * trims to the newest visible rows; max_rows == 0 means no extra limit. */
int ds4_dflash_hidden_history_copy_visible(const ds4_dflash_hidden_history *h,
                                           uint32_t anchor_position,
                                           uint32_t max_rows,
                                           float *target_hidden,
                                           uint32_t *target_positions,
                                           uint32_t *out_rows,
                                           char *err,
                                           size_t errlen);

int ds4_dflash_config_load(ds4_dflash_config *cfg,
                           const char *path,
                           char *err,
                           size_t errlen);

int ds4_dflash_config_validate_target(const ds4_dflash_config *cfg,
                                      uint32_t target_hidden_size,
                                      uint32_t target_vocab_size,
                                      uint32_t target_n_layer,
                                      char *err,
                                      size_t errlen);

int ds4_dflash_weights_validate(ds4_dflash_weights *w,
                                const char *path,
                                const ds4_dflash_config *cfg,
                                char *err,
                                size_t errlen);

int ds4_dflash_weights_open(ds4_dflash_weights *w,
                            const char *path,
                            const ds4_dflash_config *cfg,
                            char *err,
                            size_t errlen);

const ds4_dflash_tensor *ds4_dflash_weights_find_tensor(const ds4_dflash_weights *w,
                                                        const char *name);

int ds4_dflash_tensor_read_bf16_f32(const ds4_dflash_weights *w,
                                    const ds4_dflash_tensor *tensor,
                                    uint64_t elem_offset,
                                    float *out,
                                    uint64_t n,
                                    char *err,
                                    size_t errlen);

int ds4_dflash_weights_read_bf16_f32(const ds4_dflash_weights *w,
                                     const char *name,
                                     uint64_t elem_offset,
                                     float *out,
                                     uint64_t n,
                                     char *err,
                                     size_t errlen);

int ds4_dflash_project_target_hidden(const ds4_dflash_weights *w,
                                     const ds4_dflash_config *cfg,
                                     const float *tap_hc,
                                     uint32_t n_tokens,
                                     uint32_t token_index,
                                     float *target_hidden,
                                     char *err,
                                     size_t errlen);

int ds4_dflash_prepare_noise_inputs(const ds4_dflash_weights *w,
                                    const ds4_dflash_config *cfg,
                                    uint32_t anchor_token,
                                    float *noise_embedding,
                                    char *err,
                                    size_t errlen);

int ds4_dflash_prepare_block_inputs(const ds4_dflash_weights *w,
                                    const ds4_dflash_config *cfg,
                                    const float *tap_hc,
                                    uint32_t n_tokens,
                                    uint32_t token_index,
                                    uint32_t anchor_token,
                                    float *target_hidden,
                                    float *noise_embedding,
                                    char *err,
                                    size_t errlen);

int ds4_dflash_cpu_eval_mlp(const ds4_dflash_weights *w,
                            const ds4_dflash_config *cfg,
                            uint32_t layer,
                            const float *hidden_states,
                            uint32_t n_rows,
                            float *out,
                            char *err,
                            size_t errlen);

/* Single-anchor DFlash attention. target_hidden must already be filtered to
 * the base-prefix rows visible to this anchor; noise_hidden is one synthetic
 * block and attends bidirectionally within that block. */
int ds4_dflash_cpu_eval_attention(const ds4_dflash_weights *w,
                                  const ds4_dflash_config *cfg,
                                  uint32_t layer,
                                  const float *target_hidden,
                                  const uint32_t *target_positions,
                                  uint32_t n_target_rows,
                                  const float *noise_hidden,
                                  const uint32_t *noise_positions,
                                  uint32_t n_noise_rows,
                                  float *out,
                                  char *err,
                                  size_t errlen);

int ds4_dflash_cpu_eval_layer(const ds4_dflash_weights *w,
                              const ds4_dflash_config *cfg,
                              uint32_t layer,
                              const float *target_hidden,
                              const uint32_t *target_positions,
                              uint32_t n_target_rows,
                              const float *noise_hidden,
                              const uint32_t *noise_positions,
                              uint32_t n_noise_rows,
                              float *out,
                              char *err,
                              size_t errlen);

int ds4_dflash_cpu_eval_block(const ds4_dflash_weights *w,
                              const ds4_dflash_config *cfg,
                              const float *target_hidden,
                              const uint32_t *target_positions,
                              uint32_t n_target_rows,
                              const float *noise_hidden,
                              const uint32_t *noise_positions,
                              uint32_t n_noise_rows,
                              float *out,
                              char *err,
                              size_t errlen);

int ds4_dflash_cpu_eval_logits(const ds4_dflash_weights *w,
                               const ds4_dflash_config *cfg,
                               const float *hidden_states,
                               uint32_t n_rows,
                               float *logits,
                               char *err,
                               size_t errlen);

int ds4_dflash_cpu_select_tokens(const ds4_dflash_weights *w,
                                 const ds4_dflash_config *cfg,
                                 const float *logits,
                                 uint32_t n_rows,
                                 uint32_t *draft_tokens,
                                 uint32_t *target_tokens,
                                 char *err,
                                 size_t errlen);

/* Select generated draft suffix tokens from a DFlash synthetic block. Row 0 is
 * the accepted anchor position and is intentionally skipped. */
int ds4_dflash_cpu_select_draft_suffix_tokens(const ds4_dflash_weights *w,
                                              const ds4_dflash_config *cfg,
                                              const float *logits,
                                              uint32_t n_rows,
                                              uint32_t draft_cap,
                                              uint32_t *draft_tokens,
                                              uint32_t *target_tokens,
                                              char *err,
                                              size_t errlen);

#endif
