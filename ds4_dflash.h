#ifndef DS4_DFLASH_H
#define DS4_DFLASH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_DFLASH_MAX_TARGET_LAYERS 128u
#define DS4_DFLASH_MAX_PATH 4096u
#define DS4_DFLASH_MAX_TENSOR_NAME 160u
#define DS4_DSPARK_MAX_SHARDS 8u
#define DS4_DSPARK_MAX_ACTIVATED_EXPERTS 16u

typedef enum {
    DS4_DFLASH_TENSOR_UNKNOWN = 0,
    DS4_DFLASH_TENSOR_BF16,
    DS4_DFLASH_TENSOR_F32,
    DS4_DFLASH_TENSOR_F8_E4M3,
    DS4_DFLASH_TENSOR_F8_E8M0,
    DS4_DFLASH_TENSOR_I8,
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
    uint32_t shard_index;
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
    char source_path[DS4_DFLASH_MAX_PATH];
    uint32_t block_size;
    uint32_t noise_token_id;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t markov_rank;
    uint32_t sliding_window;
    uint32_t hc_mult;
    uint32_t hc_sinkhorn_iters;
    float hc_eps;
    uint32_t n_mtp_stages;
    uint32_t num_attention_heads;
    uint32_t q_lora_rank;
    uint32_t o_lora_rank;
    uint32_t head_dim;
    uint32_t qk_rope_head_dim;
    uint32_t o_groups;
    uint32_t n_routed_experts;
    uint32_t n_shared_experts;
    uint32_t n_activated_experts;
    uint32_t moe_intermediate_size;
    uint32_t rope_original_max_position_embeddings;
    float rope_theta;
    float compress_rope_theta;
    float rope_factor;
    float rope_beta_fast;
    float rope_beta_slow;
    float routed_scaling_factor;
    float swiglu_limit;
    uint32_t target_layer_ids[DS4_DFLASH_MAX_TARGET_LAYERS];
    uint32_t n_target_layer_ids;
    bool loaded;
} ds4_dspark_config;

typedef struct {
    char source_path[DS4_DFLASH_MAX_PATH];
    uint64_t header_len;
    uint64_t file_size;
    uint64_t data_start;
    void *map;
    int fd;
    bool loaded;
} ds4_dspark_shard;

typedef struct {
    char source_path[DS4_DFLASH_MAX_PATH];
    char index_path[DS4_DFLASH_MAX_PATH];
    ds4_dspark_shard shards[DS4_DSPARK_MAX_SHARDS];
    uint32_t n_shards;
    ds4_dflash_tensor *tensors;
    uint32_t n_tensors;
    uint32_t n_bound_tensors;
    bool loaded;
} ds4_dspark_weights;

typedef struct {
    uint32_t stage;
    uint32_t n_routes;
    uint32_t route_indices[DS4_DSPARK_MAX_ACTIVATED_EXPERTS];
    float route_weights[DS4_DSPARK_MAX_ACTIVATED_EXPERTS];
    double attn_hc_checksum;
    double ffn_hc_checksum;
    double ffn_out_checksum;
    double routed_sum_checksum;
    double shared_out_checksum;
    float ffn_hc_first[4];
} ds4_dspark_stage_stats;

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
void ds4_dspark_config_init(ds4_dspark_config *cfg);
void ds4_dspark_config_free(ds4_dspark_config *cfg);
void ds4_dspark_weights_init(ds4_dspark_weights *w);
void ds4_dspark_weights_free(ds4_dspark_weights *w);
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

int ds4_dflash_hidden_history_reserve_raw(ds4_dflash_hidden_history *h,
                                          uint32_t hidden_size,
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

int ds4_dspark_config_load(ds4_dspark_config *cfg,
                           const char *path,
                           char *err,
                           size_t errlen);

int ds4_dspark_config_validate_target(const ds4_dspark_config *cfg,
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

int ds4_dspark_weights_open(ds4_dspark_weights *w,
                            const char *path,
                            const ds4_dspark_config *cfg,
                            char *err,
                            size_t errlen);

int ds4_dspark_weights_open_graph(ds4_dspark_weights *w,
                                  const char *path,
                                  const ds4_dspark_config *cfg,
                                  char *err,
                                  size_t errlen);

int ds4_dspark_weights_validate(ds4_dspark_weights *w,
                                const char *path,
                                const ds4_dspark_config *cfg,
                                char *err,
                                size_t errlen);

const ds4_dflash_tensor *ds4_dspark_weights_find_tensor(const ds4_dspark_weights *w,
                                                        const char *name);

float ds4_dflash_fp8_e4m3_to_f32(uint8_t raw);
float ds4_dflash_fp8_e8m0_to_f32(uint8_t raw);
float ds4_dflash_fp4_e2m1_to_f32(uint8_t raw);

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

int ds4_dspark_tensor_read_bf16_f32(const ds4_dspark_weights *w,
                                    const ds4_dflash_tensor *tensor,
                                    uint64_t elem_offset,
                                    float *out,
                                    uint64_t n,
                                    char *err,
                                    size_t errlen);

int ds4_dspark_weights_read_bf16_f32(const ds4_dspark_weights *w,
                                     const char *name,
                                     uint64_t elem_offset,
                                     float *out,
                                     uint64_t n,
                                     char *err,
                                     size_t errlen);

int ds4_dspark_tensor_read_f32(const ds4_dspark_weights *w,
                               const ds4_dflash_tensor *tensor,
                               uint64_t elem_offset,
                               float *out,
                               uint64_t n,
                               char *err,
                               size_t errlen);

int ds4_dspark_weights_read_f32(const ds4_dspark_weights *w,
                                const char *name,
                                uint64_t elem_offset,
                                float *out,
                                uint64_t n,
                                char *err,
                                size_t errlen);

int ds4_dspark_linear_f8_f32(const ds4_dspark_weights *w,
                             const ds4_dflash_tensor *weight,
                             const ds4_dflash_tensor *scale,
                             const float *in,
                             float *out,
                             char *err,
                             size_t errlen);

int ds4_dspark_grouped_linear_f8_f32(const ds4_dspark_weights *w,
                                     const ds4_dflash_tensor *weight,
                                     const ds4_dflash_tensor *scale,
                                     const float *in,
                                     uint32_t groups,
                                     float *out,
                                     char *err,
                                     size_t errlen);

int ds4_dspark_linear_bf16_f32(const ds4_dspark_weights *w,
                               const ds4_dflash_tensor *weight,
                               const float *in,
                               float *out,
                               char *err,
                               size_t errlen);

int ds4_dspark_linear_fp4_f32(const ds4_dspark_weights *w,
                              const ds4_dflash_tensor *weight,
                              const ds4_dflash_tensor *scale,
                              const float *in,
                              float *out,
                              char *err,
                              size_t errlen);

int ds4_dspark_rms_norm_bf16(const ds4_dspark_weights *w,
                             const ds4_dflash_tensor *weight,
                             const float *in,
                             uint64_t n,
                             float *out,
                             char *err,
                             size_t errlen);

int ds4_dspark_rms_norm_f32(const float *in,
                            uint64_t n,
                            float *out,
                            char *err,
                            size_t errlen);

int ds4_dspark_hc_collapse_f32(const ds4_dspark_weights *w,
                               const ds4_dflash_tensor *fn,
                               const ds4_dflash_tensor *scale,
                               const ds4_dflash_tensor *base,
                               const float *residual_hc,
                               uint32_t hidden,
                               uint32_t hc_mult,
                               uint32_t sinkhorn_iters,
                               float eps,
                               float *collapsed,
                               float *split,
                               char *err,
                               size_t errlen);

int ds4_dspark_hc_expand_f32(const float *block_out,
                             const float *residual_hc,
                             const float *split,
                             uint32_t hidden,
                             uint32_t hc_mult,
                             float *out_hc,
                             char *err,
                             size_t errlen);

int ds4_dspark_apply_partial_rope_f32(float *x,
                                      uint32_t n_heads,
                                      uint32_t head_dim,
                                      uint32_t rope_dim,
                                      uint64_t position,
                                      float theta,
                                      uint32_t original_seq_len,
                                      float factor,
                                      float beta_fast,
                                      float beta_slow,
                                      bool inverse,
                                      char *err,
                                      size_t errlen);

int ds4_dspark_sparse_attention_one_f32(const ds4_dspark_weights *w,
                                        const ds4_dflash_tensor *sink,
                                        const float *q,
                                        const float *kv,
                                        uint32_t n_heads,
                                        uint32_t head_dim,
                                        float *out,
                                        char *err,
                                        size_t errlen);

int ds4_dspark_sparse_attention_block_f32(const ds4_dspark_weights *w,
                                          const ds4_dflash_tensor *sink,
                                          const float *q,
                                          const float *kv,
                                          uint32_t q_len,
                                          uint32_t kv_len,
                                          uint32_t n_heads,
                                          uint32_t head_dim,
                                          float *out,
                                          char *err,
                                          size_t errlen);

int ds4_dspark_moe_gate_topk_f32(const ds4_dspark_weights *w,
                                 const ds4_dflash_tensor *weight,
                                 const ds4_dflash_tensor *bias,
                                 const float *in,
                                 uint32_t topk,
                                 float route_scale,
                                 uint32_t *indices,
                                 float *weights,
                                 char *err,
                                 size_t errlen);

int ds4_dspark_swiglu_f32(const float *gate,
                          const float *up,
                          uint64_t n,
                          float limit,
                          float *out,
                          char *err,
                          size_t errlen);

void ds4_dspark_stage_stats_init(ds4_dspark_stage_stats *stats);

int ds4_dspark_run_stage_reference(const ds4_dspark_weights *w,
                                   const ds4_dspark_config *cfg,
                                   uint32_t stage,
                                   uint64_t position,
                                   const float *in_hc,
                                   float *out_hc,
                                   ds4_dspark_stage_stats *stats,
                                   char *err,
                                   size_t errlen);

int ds4_dspark_run_stage_block_reference(const ds4_dspark_weights *w,
                                         const ds4_dspark_config *cfg,
                                         uint32_t stage,
                                         uint64_t position0,
                                         const float *in_hc_rows,
                                         uint32_t n_rows,
                                         float *out_hc_rows,
                                         char *err,
                                         size_t errlen);

int ds4_dspark_run_stage_block_context_reference(const ds4_dspark_weights *w,
                                                 const ds4_dspark_config *cfg,
                                                 uint32_t stage,
                                                 const float *context_main_rows,
                                                 const uint32_t *context_positions,
                                                 uint32_t context_rows,
                                                 uint64_t position0,
                                                 const float *in_hc_rows,
                                                 uint32_t n_rows,
                                                 float *out_hc_rows,
                                                 char *err,
                                                 size_t errlen);

int ds4_dspark_init_hc_block_from_main_f32(const ds4_dspark_config *cfg,
                                           const float *main_x,
                                           uint32_t n_rows,
                                           float *out_hc_rows,
                                           char *err,
                                           size_t errlen);

int ds4_dspark_project_main_hidden(const ds4_dspark_weights *w,
                                   const ds4_dspark_config *cfg,
                                   const float *main_hidden,
                                   float *main_x,
                                   char *err,
                                   size_t errlen);

int ds4_dspark_final_hc_head_f32(const ds4_dspark_weights *w,
                                 const ds4_dspark_config *cfg,
                                 const float *in_hc,
                                 float *out,
                                 char *err,
                                 size_t errlen);

int ds4_dspark_final_norm_f32(const ds4_dspark_weights *w,
                              const ds4_dspark_config *cfg,
                              const float *in,
                              float *out,
                              char *err,
                              size_t errlen);

int ds4_dspark_final_block_norm_f32(const ds4_dspark_weights *w,
                                    const ds4_dspark_config *cfg,
                                    const float *in_hc_rows,
                                    uint32_t n_rows,
                                    float *hidden_rows,
                                    float *normed_rows,
                                    char *err,
                                    size_t errlen);

int ds4_dspark_markov_prev_embedding_f32(const ds4_dspark_weights *w,
                                         const ds4_dspark_config *cfg,
                                         uint32_t prev_token_id,
                                         float *embedding,
                                         char *err,
                                         size_t errlen);

int ds4_dspark_markov_logits_f32(const ds4_dspark_weights *w,
                                 const ds4_dspark_config *cfg,
                                 const float *prev_embedding,
                                 float *logits,
                                 char *err,
                                 size_t errlen);

int ds4_dspark_markov_logits_subset_f32(const ds4_dspark_weights *w,
                                        const ds4_dspark_config *cfg,
                                        const float *prev_embedding,
                                        const uint32_t *token_ids,
                                        uint32_t n_tokens,
                                        float *logits,
                                        char *err,
                                        size_t errlen);

int ds4_dspark_confidence_logit_f32(const ds4_dspark_weights *w,
                                    const ds4_dspark_config *cfg,
                                    const float *hidden,
                                    const float *prev_embedding,
                                    float *logit,
                                    char *err,
                                    size_t errlen);

int ds4_dspark_select_draft_tokens_argmax(const ds4_dspark_weights *w,
                                          const ds4_dspark_config *cfg,
                                          const float *base_logits,
                                          const float *hidden_rows,
                                          uint32_t proposal_len,
                                          uint32_t first_prev_token_id,
                                          float confidence_threshold,
                                          uint32_t *draft_tokens,
                                          float *margins,
                                          float *confidence_logits,
                                          char *err,
                                          size_t errlen);

int ds4_dspark_select_draft_tokens_topk_argmax(const ds4_dspark_weights *w,
                                               const ds4_dspark_config *cfg,
                                               const uint32_t *topk_token_ids,
                                               const float *topk_base_logits,
                                               uint32_t proposal_len,
                                               uint32_t top_k,
                                               uint32_t first_prev_token_id,
                                               uint32_t *draft_tokens,
                                               float *margins,
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
                                              float *margins,
                                              char *err,
                                              size_t errlen);

#endif
