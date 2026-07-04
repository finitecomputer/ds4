#include "ds4_dflash.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int failures;
static char temp_root[PATH_MAX];

#define EXPECT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            failures++; \
        } \
    } while (0)

#define EXPECT_NEAR(a, b, tol) \
    do { \
        if (fabs((double)(a) - (double)(b)) > (double)(tol)) { \
            fprintf(stderr, "FAIL %s:%d: %s ~= %s (got %.9g vs %.9g)\n", \
                    __FILE__, __LINE__, #a, #b, (double)(a), (double)(b)); \
            failures++; \
        } \
    } while (0)

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} test_buf;

static void buf_appendf(test_buf *b, const char *fmt, ...) {
    va_list ap;
    int n = 0;

    if (b->len >= b->cap) abort();
    va_start(ap, fmt);
    n = vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= b->cap - b->len) abort();
    b->len += (size_t)n;
}

static void write_le64(FILE *fp, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)((v >> (8 * i)) & 0xffu);
    if (fwrite(b, 1, sizeof(b), fp) != sizeof(b)) {
        perror("fwrite");
        abort();
    }
}

static void write_le16_at(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
}

static void write_le32_at(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
    p[2] = (unsigned char)((v >> 16) & 0xffu);
    p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static void write_le64_at(unsigned char *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)((v >> (8 * i)) & 0xffu);
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float test_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void write_bf16(unsigned char *data, uint64_t base, uint64_t elem, float value) {
    write_le16_at(data + base + elem * 2u, f32_to_bf16(value));
}

static void write_f32(unsigned char *data, uint64_t base, uint64_t elem, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    write_le32_at(data + base + elem * 4u, bits);
}

static void expected_tiny_mlp_row(const float *hidden, float *expected) {
    const float mean_square =
        (hidden[0] * hidden[0] +
         hidden[1] * hidden[1] +
         hidden[2] * hidden[2] +
         hidden[3] * hidden[3]) / 4.0f;
    const float inv_rms = 1.0f / sqrtf(mean_square + 1.0e-6f);
    const float h0 = hidden[0] * inv_rms;
    const float h1 = hidden[1] * inv_rms;
    const float h2 = hidden[2] * inv_rms;
    const float m0 = test_silu(h0) * (2.0f * h0);
    const float m1 = test_silu(h1) * (3.0f * h1);
    const float m2 = test_silu(h2) * (4.0f * h2);

    expected[0] = hidden[0] + m0;
    expected[1] = hidden[1] + m1;
    expected[2] = hidden[2] + m2;
    expected[3] = hidden[3] + m0 + m1 + m2;
}

static void make_temp_root(void) {
    for (int i = 0; i < 100; i++) {
        int n = snprintf(temp_root,
                         sizeof(temp_root),
                         "/tmp/ds4-dflash-config-test-%ld-%d",
                         (long)getpid(),
                         i);
        if (n < 0 || (size_t)n >= sizeof(temp_root)) abort();
        if (mkdir(temp_root, 0700) == 0) return;
        if (errno != EEXIST) {
            perror("mkdir");
            abort();
        }
    }
    abort();
}

static void write_config(const char *name,
                         const char *json,
                         char *path,
                         size_t pathlen) {
    int n = snprintf(path, pathlen, "%s/%s", temp_root, name);
    FILE *fp = NULL;

    if (n < 0 || (size_t)n >= pathlen) abort();
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        abort();
    }
    if (fwrite(json, 1, strlen(json), fp) != strlen(json)) {
        perror("fwrite");
        abort();
    }
    fclose(fp);
}

static const char *real_deepseek_dflash_json(void) {
    return "{\n"
           "  \"architectures\": [\"DFlashDraftModel\"],\n"
           "  \"speculators_model_type\": \"dflash\",\n"
           "  \"block_size\": 8,\n"
           "  \"draft_vocab_size\": 32000,\n"
           "  \"mask_token_id\": 1,\n"
           "  \"max_anchors\": 3072,\n"
           "  \"sliding_window_non_causal\": false,\n"
           "  \"target_hidden_size\": null,\n"
           "  \"aux_hidden_state_layer_ids\": [3, 13, 23, 32, 42],\n"
           "  \"transformer_layer_config\": {\n"
           "    \"hidden_size\": 4096,\n"
           "    \"vocab_size\": 129280,\n"
           "    \"num_hidden_layers\": 5,\n"
           "    \"num_attention_heads\": 64,\n"
           "    \"num_key_value_heads\": 1,\n"
           "    \"head_dim\": 256,\n"
           "    \"intermediate_size\": 2048,\n"
           "    \"hc_mult\": 4,\n"
           "    \"rope_parameters\": {\"rope_theta\": 10000, \"rope_type\": \"default\"},\n"
           "    \"sliding_window\": 2048\n"
           "  },\n"
           "  \"sliding_window\": 16\n"
           "}\n";
}

static const char *legacy_deepseek_shape_json(void) {
    return "{\n"
           "  \"architectures\": [\"DFlashDraftModel\"],\n"
           "  \"block_size\": 16,\n"
           "  \"hidden_size\": 4096,\n"
           "  \"vocab_size\": 129280,\n"
           "  \"num_target_layers\": 43,\n"
           "  \"num_hidden_layers\": 5,\n"
           "  \"dflash_config\": {\n"
           "    \"mask_token_id\": 129279,\n"
           "    \"target_layer_ids\": [1, 10, 20, 30, 42]\n"
           "  }\n"
           "}\n";
}

static void append_tensor1(test_buf *b, bool *first, const char *name,
                           const char *dtype, uint64_t d0) {
    buf_appendf(b,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu],\"data_offsets\":[0,0]}",
                *first ? "" : ",",
                name,
                dtype,
                (unsigned long long)d0);
    *first = false;
}

static void append_tensor2(test_buf *b, bool *first, const char *name,
                           const char *dtype, uint64_t d0, uint64_t d1) {
    buf_appendf(b,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],\"data_offsets\":[0,0]}",
                *first ? "" : ",",
                name,
                dtype,
                (unsigned long long)d0,
                (unsigned long long)d1);
    *first = false;
}

static uint64_t tiny_dtype_size(const char *dtype) {
    if (!strcmp(dtype, "BF16")) return 2;
    if (!strcmp(dtype, "F32")) return 4;
    if (!strcmp(dtype, "F8_E4M3")) return 1;
    if (!strcmp(dtype, "F8_E8M0")) return 1;
    if (!strcmp(dtype, "I8")) return 1;
    if (!strcmp(dtype, "I64")) return 8;
    if (!strcmp(dtype, "BOOL")) return 1;
    abort();
}

static void append_tiny_tensor1(test_buf *b, bool *first, const char *name,
                                const char *dtype, uint64_t d0, uint64_t *off) {
    const uint64_t start = *off;
    const uint64_t bytes = d0 * tiny_dtype_size(dtype);
    *off += bytes;
    buf_appendf(b,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu],\"data_offsets\":[%llu,%llu]}",
                *first ? "" : ",",
                name,
                dtype,
                (unsigned long long)d0,
                (unsigned long long)start,
                (unsigned long long)*off);
    *first = false;
}

static void append_tiny_tensor2(test_buf *b, bool *first, const char *name,
                                const char *dtype, uint64_t d0, uint64_t d1,
                                uint64_t *off, uint64_t *start_out) {
    const uint64_t start = *off;
    const uint64_t bytes = d0 * d1 * tiny_dtype_size(dtype);
    *off += bytes;
    if (start_out) *start_out = start;
    buf_appendf(b,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],\"data_offsets\":[%llu,%llu]}",
                *first ? "" : ",",
                name,
                dtype,
                (unsigned long long)d0,
                (unsigned long long)d1,
                (unsigned long long)start,
                (unsigned long long)*off);
    *first = false;
}

static void write_safetensors_fixture(bool bad_fc_shape) {
    char path[PATH_MAX];
    char *header = calloc(1, 65536);
    test_buf b = {.ptr = header, .cap = 65536};
    bool first = true;
    FILE *fp = NULL;

    if (!header) abort();
    buf_appendf(&b, "{");
    append_tensor1(&b, &first, "d2t", "I64", 32000);
    append_tensor1(&b, &first, "t2d", "BOOL", 129280);
    append_tensor2(&b, &first, "embed_tokens.weight", "BF16", 129280, 4096);
    append_tensor2(&b, &first, "fc.weight", "BF16", 4096,
                   bad_fc_shape ? 81919 : 81920);
    append_tensor1(&b, &first, "hidden_norm.weight", "BF16", 4096);
    append_tensor1(&b, &first, "norm.weight", "BF16", 4096);
    append_tensor2(&b, &first, "lm_head.weight", "BF16", 32000, 4096);
    for (int il = 0; il < 5; il++) {
        char name[160];
        snprintf(name, sizeof(name), "layers.%d.input_layernorm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 4096);
        snprintf(name, sizeof(name), "layers.%d.post_attention_layernorm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 4096);
        snprintf(name, sizeof(name), "layers.%d.mlp.gate_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 2048, 4096);
        snprintf(name, sizeof(name), "layers.%d.mlp.up_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 2048, 4096);
        snprintf(name, sizeof(name), "layers.%d.mlp.down_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 4096, 2048);
        snprintf(name, sizeof(name), "layers.%d.self_attn.q_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 16384, 4096);
        snprintf(name, sizeof(name), "layers.%d.self_attn.k_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 256, 4096);
        snprintf(name, sizeof(name), "layers.%d.self_attn.v_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 256, 4096);
        snprintf(name, sizeof(name), "layers.%d.self_attn.o_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 4096, 16384);
        snprintf(name, sizeof(name), "layers.%d.self_attn.q_norm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 256);
        snprintf(name, sizeof(name), "layers.%d.self_attn.k_norm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 256);
    }
    buf_appendf(&b, "}");

    if (snprintf(path, sizeof(path), "%s/model.safetensors", temp_root) < 0) abort();
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        abort();
    }
    write_le64(fp, (uint64_t)b.len);
    if (fwrite(header, 1, b.len, fp) != b.len) {
        perror("fwrite");
        abort();
    }
    fclose(fp);
    free(header);
}

static void write_safetensors_fixture_with_fc_dtype(const char *fc_dtype) {
    char path[PATH_MAX];
    char *header = calloc(1, 65536);
    test_buf b = {.ptr = header, .cap = 65536};
    bool first = true;
    FILE *fp = NULL;

    if (!header) abort();
    buf_appendf(&b, "{");
    append_tensor1(&b, &first, "d2t", "I64", 32000);
    append_tensor1(&b, &first, "t2d", "BOOL", 129280);
    append_tensor2(&b, &first, "embed_tokens.weight", "BF16", 129280, 4096);
    append_tensor2(&b, &first, "fc.weight", fc_dtype, 4096, 81920);
    append_tensor1(&b, &first, "hidden_norm.weight", "BF16", 4096);
    append_tensor1(&b, &first, "norm.weight", "BF16", 4096);
    append_tensor2(&b, &first, "lm_head.weight", "BF16", 32000, 4096);
    for (int il = 0; il < 5; il++) {
        char name[160];
        snprintf(name, sizeof(name), "layers.%d.input_layernorm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 4096);
        snprintf(name, sizeof(name), "layers.%d.post_attention_layernorm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 4096);
        snprintf(name, sizeof(name), "layers.%d.mlp.gate_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 2048, 4096);
        snprintf(name, sizeof(name), "layers.%d.mlp.up_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 2048, 4096);
        snprintf(name, sizeof(name), "layers.%d.mlp.down_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 4096, 2048);
        snprintf(name, sizeof(name), "layers.%d.self_attn.q_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 16384, 4096);
        snprintf(name, sizeof(name), "layers.%d.self_attn.k_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 256, 4096);
        snprintf(name, sizeof(name), "layers.%d.self_attn.v_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 256, 4096);
        snprintf(name, sizeof(name), "layers.%d.self_attn.o_proj.weight", il);
        append_tensor2(&b, &first, name, "BF16", 4096, 16384);
        snprintf(name, sizeof(name), "layers.%d.self_attn.q_norm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 256);
        snprintf(name, sizeof(name), "layers.%d.self_attn.k_norm.weight", il);
        append_tensor1(&b, &first, name, "BF16", 256);
    }
    buf_appendf(&b, "}");

    if (snprintf(path, sizeof(path), "%s/model.safetensors", temp_root) < 0) abort();
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        abort();
    }
    write_le64(fp, (uint64_t)b.len);
    if (fwrite(header, 1, b.len, fp) != b.len) {
        perror("fwrite");
        abort();
    }
    fclose(fp);
    free(header);
}

static void write_tiny_safetensors_fixture_with_mapping(bool disable_target7,
                                                        bool draft3_out_of_range) {
    char path[PATH_MAX];
    char *header = calloc(1, 65536);
    test_buf b = {.ptr = header, .cap = 65536};
    bool first = true;
    uint64_t off = 0;
    uint64_t d2t_off = 0;
    uint64_t t2d_off = 0;
    uint64_t embed_off = 0;
    uint64_t fc_off = 0;
    uint64_t hidden_norm_off = 0;
    uint64_t norm_off = 0;
    uint64_t lm_head_off = 0;
    uint64_t input_norm_off = 0;
    uint64_t post_norm_off = 0;
    uint64_t gate_off = 0;
    uint64_t up_off = 0;
    uint64_t down_off = 0;
    uint64_t q_proj_off = 0;
    uint64_t k_proj_off = 0;
    uint64_t v_proj_off = 0;
    uint64_t o_proj_off = 0;
    uint64_t q_norm_off = 0;
    uint64_t k_norm_off = 0;
    FILE *fp = NULL;

    if (!header) abort();
    buf_appendf(&b, "{");
    d2t_off = off;
    append_tiny_tensor1(&b, &first, "d2t", "I64", 4, &off);
    t2d_off = off;
    append_tiny_tensor1(&b, &first, "t2d", "BOOL", 8, &off);
    append_tiny_tensor2(&b, &first, "embed_tokens.weight", "BF16", 8, 4, &off, &embed_off);
    append_tiny_tensor2(&b, &first, "fc.weight", "BF16", 4, 8, &off, &fc_off);
    hidden_norm_off = off;
    append_tiny_tensor1(&b, &first, "hidden_norm.weight", "BF16", 4, &off);
    norm_off = off;
    append_tiny_tensor1(&b, &first, "norm.weight", "BF16", 4, &off);
    append_tiny_tensor2(&b, &first, "lm_head.weight", "BF16", 4, 4, &off, &lm_head_off);
    input_norm_off = off;
    append_tiny_tensor1(&b, &first, "layers.0.input_layernorm.weight", "BF16", 4, &off);
    post_norm_off = off;
    append_tiny_tensor1(&b, &first, "layers.0.post_attention_layernorm.weight", "BF16", 4, &off);
    append_tiny_tensor2(&b, &first, "layers.0.mlp.gate_proj.weight", "BF16", 3, 4, &off, &gate_off);
    append_tiny_tensor2(&b, &first, "layers.0.mlp.up_proj.weight", "BF16", 3, 4, &off, &up_off);
    append_tiny_tensor2(&b, &first, "layers.0.mlp.down_proj.weight", "BF16", 4, 3, &off, &down_off);
    append_tiny_tensor2(&b, &first, "layers.0.self_attn.q_proj.weight", "BF16", 4, 4, &off, &q_proj_off);
    append_tiny_tensor2(&b, &first, "layers.0.self_attn.k_proj.weight", "BF16", 2, 4, &off, &k_proj_off);
    append_tiny_tensor2(&b, &first, "layers.0.self_attn.v_proj.weight", "BF16", 2, 4, &off, &v_proj_off);
    append_tiny_tensor2(&b, &first, "layers.0.self_attn.o_proj.weight", "BF16", 4, 4, &off, &o_proj_off);
    q_norm_off = off;
    append_tiny_tensor1(&b, &first, "layers.0.self_attn.q_norm.weight", "BF16", 2, &off);
    k_norm_off = off;
    append_tiny_tensor1(&b, &first, "layers.0.self_attn.k_norm.weight", "BF16", 2, &off);
    buf_appendf(&b, "}");

    unsigned char *data = calloc(1, (size_t)off);
    if (!data) abort();
    /* DFlash stores d2t as an offset from draft id to target id. */
    write_le64_at(data + d2t_off + 0u * 8u, 2);
    write_le64_at(data + d2t_off + 1u * 8u, 2);
    write_le64_at(data + d2t_off + 2u * 8u, 3);
    write_le64_at(data + d2t_off + 3u * 8u, draft3_out_of_range ? 96 : 4);
    data[t2d_off + 2] = 1;
    data[t2d_off + 3] = 1;
    data[t2d_off + 5] = 1;
    data[t2d_off + 7] = disable_target7 ? 0 : 1;
    for (int i = 0; i < 4; i++) {
        write_bf16(data, embed_off, (uint64_t)(4 + i), (float)(i + 1));
        write_bf16(data, fc_off, (uint64_t)(i * 8 + i), 1.0f);
        write_bf16(data, hidden_norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, input_norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, post_norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, q_proj_off, (uint64_t)(i * 4 + i), 1.0f);
        write_bf16(data, o_proj_off, (uint64_t)(i * 4 + i), 1.0f);
    }
    write_bf16(data, lm_head_off, 4, 1.0f);
    write_bf16(data, lm_head_off, 9, 2.0f);
    write_bf16(data, lm_head_off, 14, 3.0f);
    write_bf16(data, k_proj_off, 0, 1.0f);
    write_bf16(data, k_proj_off, 5, 1.0f);
    write_bf16(data, v_proj_off, 0, 1.0f);
    write_bf16(data, v_proj_off, 5, 1.0f);
    write_bf16(data, q_norm_off, 0, 1.0f);
    write_bf16(data, q_norm_off, 1, 1.0f);
    write_bf16(data, k_norm_off, 0, 1.0f);
    write_bf16(data, k_norm_off, 1, 1.0f);
    write_bf16(data, gate_off, 0, 1.0f);
    write_bf16(data, gate_off, 5, 1.0f);
    write_bf16(data, gate_off, 10, 1.0f);
    write_bf16(data, up_off, 0, 2.0f);
    write_bf16(data, up_off, 5, 3.0f);
    write_bf16(data, up_off, 10, 4.0f);
    write_bf16(data, down_off, 0, 1.0f);
    write_bf16(data, down_off, 4, 1.0f);
    write_bf16(data, down_off, 8, 1.0f);
    write_bf16(data, down_off, 9, 1.0f);
    write_bf16(data, down_off, 10, 1.0f);
    write_bf16(data, down_off, 11, 1.0f);

    if (snprintf(path, sizeof(path), "%s/model.safetensors", temp_root) < 0) abort();
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        abort();
    }
    write_le64(fp, (uint64_t)b.len);
    if (fwrite(header, 1, b.len, fp) != b.len ||
        fwrite(data, 1, (size_t)off, fp) != (size_t)off) {
        perror("fwrite");
        abort();
    }
    fclose(fp);
    free(data);
    free(header);
}

static void write_tiny_safetensors_fixture(void) {
    write_tiny_safetensors_fixture_with_mapping(false, false);
}

static const char *tiny_dspark_json(void) {
    return "{\n"
           "  \"architectures\": [\"Transformer\"],\n"
           "  \"hidden_size\": 4,\n"
           "  \"vocab_size\": 8,\n"
           "  \"dspark_block_size\": 2,\n"
           "  \"dspark_noise_token_id\": 7,\n"
           "  \"dspark_target_layer_ids\": [1, 2],\n"
           "  \"dspark_markov_rank\": 3,\n"
           "  \"sliding_window\": 4,\n"
           "  \"hc_mult\": 2\n"
           "}\n";
}

static const char *tiny_dspark_index_json(void) {
    return "{\n"
           "  \"metadata\": {\"total_size\": 0},\n"
           "  \"weight_map\": {\n"
           "    \"embed.weight\": \"model-00001-of-00002.safetensors\",\n"
           "    \"head.weight\": \"model-00002-of-00002.safetensors\",\n"
           "    \"norm.weight\": \"model-00002-of-00002.safetensors\",\n"
           "    \"hc_head_fn\": \"model-00002-of-00002.safetensors\",\n"
           "    \"hc_head_base\": \"model-00002-of-00002.safetensors\",\n"
           "    \"hc_head_scale\": \"model-00002-of-00002.safetensors\",\n"
           "    \"mtp.0.main_proj.weight\": \"dspark-mtp-00001-of-00003.safetensors\",\n"
           "    \"mtp.0.main_proj.scale\": \"dspark-mtp-00001-of-00003.safetensors\",\n"
           "    \"mtp.0.main_norm.weight\": \"dspark-mtp-00001-of-00003.safetensors\",\n"
           "    \"mtp.2.norm.weight\": \"dspark-mtp-00003-of-00003.safetensors\",\n"
           "    \"mtp.2.markov_head.markov_w1.weight\": \"dspark-mtp-00003-of-00003.safetensors\",\n"
           "    \"mtp.2.markov_head.markov_w2.weight\": \"dspark-mtp-00003-of-00003.safetensors\",\n"
           "    \"mtp.2.confidence_head.proj.weight\": \"dspark-mtp-00003-of-00003.safetensors\",\n"
           "    \"mtp.2.hc_head_fn\": \"dspark-mtp-00003-of-00003.safetensors\",\n"
           "    \"mtp.2.hc_head_base\": \"dspark-mtp-00003-of-00003.safetensors\",\n"
           "    \"mtp.2.hc_head_scale\": \"dspark-mtp-00003-of-00003.safetensors\"\n"
           "  }\n"
           "}\n";
}

static void write_tiny_dspark_shard0(void) {
    char path[PATH_MAX];
    char *header = calloc(1, 8192);
    test_buf b = {.ptr = header, .cap = 8192};
    bool first = true;
    uint64_t off = 0;
    uint64_t main_proj_off = 0;
    uint64_t main_scale_off = 0;
    uint64_t norm_off = 0;
    FILE *fp = NULL;

    if (!header) abort();
    buf_appendf(&b, "{");
    main_proj_off = off;
    append_tiny_tensor2(&b, &first, "mtp.0.main_proj.weight", "F8_E4M3", 4, 8, &off, NULL);
    main_scale_off = off;
    append_tiny_tensor2(&b, &first, "mtp.0.main_proj.scale", "F8_E8M0", 1, 1, &off, NULL);
    norm_off = off;
    append_tiny_tensor1(&b, &first, "mtp.0.main_norm.weight", "BF16", 4, &off);
    buf_appendf(&b, "}");

    unsigned char *data = calloc(1, (size_t)off);
    if (!data) abort();
    for (int i = 0; i < 4; i++) data[main_proj_off + (uint64_t)i * 8u + (uint64_t)i] = 0x38u;
    data[main_scale_off] = 127u;
    for (int i = 0; i < 4; i++) write_bf16(data, norm_off, (uint64_t)i, (float)(i + 1));

    if (snprintf(path, sizeof(path), "%s/dspark-mtp-00001-of-00003.safetensors", temp_root) < 0) abort();
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        abort();
    }
    write_le64(fp, (uint64_t)b.len);
    if (fwrite(header, 1, b.len, fp) != b.len ||
        fwrite(data, 1, (size_t)off, fp) != (size_t)off) {
        perror("fwrite");
        abort();
    }
    fclose(fp);
    free(data);
    free(header);
}

static void write_tiny_dspark_shard2(void) {
    char path[PATH_MAX];
    char *header = calloc(1, 8192);
    test_buf b = {.ptr = header, .cap = 8192};
    bool first = true;
    uint64_t off = 0;
    uint64_t norm_off = 0;
    uint64_t markov_w1_off = 0;
    uint64_t markov_w2_off = 0;
    uint64_t confidence_off = 0;
    uint64_t hc_fn_off = 0;
    uint64_t hc_base_off = 0;
    uint64_t hc_scale_off = 0;
    FILE *fp = NULL;

    if (!header) abort();
    buf_appendf(&b, "{");
    norm_off = off;
    append_tiny_tensor1(&b, &first, "mtp.2.norm.weight", "BF16", 4, &off);
    markov_w1_off = off;
    append_tiny_tensor2(&b, &first, "mtp.2.markov_head.markov_w1.weight", "BF16", 8, 3, &off, NULL);
    markov_w2_off = off;
    append_tiny_tensor2(&b, &first, "mtp.2.markov_head.markov_w2.weight", "BF16", 8, 3, &off, NULL);
    confidence_off = off;
    append_tiny_tensor2(&b, &first, "mtp.2.confidence_head.proj.weight", "BF16", 1, 7, &off, NULL);
    hc_fn_off = off;
    append_tiny_tensor2(&b, &first, "mtp.2.hc_head_fn", "F32", 2, 8, &off, NULL);
    hc_base_off = off;
    append_tiny_tensor1(&b, &first, "mtp.2.hc_head_base", "F32", 2, &off);
    hc_scale_off = off;
    append_tiny_tensor1(&b, &first, "mtp.2.hc_head_scale", "F32", 1, &off);
    buf_appendf(&b, "}");

    unsigned char *data = calloc(1, (size_t)off);
    if (!data) abort();
    for (int i = 0; i < 4; i++) write_bf16(data, norm_off, (uint64_t)i, (float)(10 + i));
    write_bf16(data, markov_w1_off, 9, 1.0f);
    write_bf16(data, markov_w1_off, 10, 2.0f);
    write_bf16(data, markov_w1_off, 11, 3.0f);
    write_bf16(data, markov_w2_off, 0, 1.0f);
    write_bf16(data, markov_w2_off, 4, 2.0f);
    write_bf16(data, markov_w2_off, 8, 3.0f);
    write_bf16(data, markov_w2_off, 9, 1.0f);
    write_bf16(data, markov_w2_off, 10, 1.0f);
    write_bf16(data, markov_w2_off, 11, 1.0f);
    for (int i = 0; i < 7; i++) write_bf16(data, confidence_off, (uint64_t)i, (float)(i + 1));
    write_f32(data, hc_fn_off, 0, 1.0f);
    write_f32(data, hc_fn_off, 12, -1.0f);
    write_f32(data, hc_base_off, 0, 0.5f);
    write_f32(data, hc_base_off, 1, 1.5f);
    write_f32(data, hc_scale_off, 0, 2.0f);

    if (snprintf(path, sizeof(path), "%s/dspark-mtp-00003-of-00003.safetensors", temp_root) < 0) abort();
    fp = fopen(path, "wb");
    if (!fp) {
        perror(path);
        abort();
    }
    write_le64(fp, (uint64_t)b.len);
    if (fwrite(header, 1, b.len, fp) != b.len ||
        fwrite(data, 1, (size_t)off, fp) != (size_t)off) {
        perror("fwrite");
        abort();
    }
    fclose(fp);
    free(data);
    free(header);
}

static void test_valid_deepseek_config_file(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("valid.json", real_deepseek_dflash_json(), path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) == 0);
    EXPECT(cfg.loaded);
    EXPECT(cfg.block_size == 8);
    EXPECT(cfg.hidden_size == 4096);
    EXPECT(cfg.target_hidden_size == 4096);
    EXPECT(cfg.vocab_size == 129280);
    EXPECT(cfg.draft_vocab_size == 32000);
    EXPECT(cfg.num_target_layers == 0);
    EXPECT(cfg.num_hidden_layers == 5);
    EXPECT(cfg.intermediate_size == 2048);
    EXPECT(cfg.num_attention_heads == 64);
    EXPECT(cfg.num_key_value_heads == 1);
    EXPECT(cfg.head_dim == 256);
    EXPECT(cfg.hc_mult == 4);
    EXPECT(cfg.sliding_window == 16);
    EXPECT(!cfg.sliding_window_non_causal);
    EXPECT_NEAR(cfg.rope_theta, 10000.0f, 0.01f);
    EXPECT(cfg.n_target_layer_ids == 5);
    EXPECT(cfg.target_layer_ids[0] == 2);
    EXPECT(cfg.target_layer_ids[4] == 41);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) == 0);
    ds4_dflash_config_free(&cfg);
}

static void test_directory_resolves_config_json(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("config.json", legacy_deepseek_shape_json(), path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(strstr(cfg.source_path, "config.json") != NULL);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) == 0);
    ds4_dflash_config_free(&cfg);
}

static void test_public_qwen_shape_is_rejected_for_deepseek(void) {
    const char *json =
        "{\n"
        "  \"architectures\": [\"DFlashDraftModel\"],\n"
        "  \"block_size\": 16,\n"
        "  \"hidden_size\": 5120,\n"
        "  \"vocab_size\": 248320,\n"
        "  \"num_target_layers\": 64,\n"
        "  \"dflash_config\": {\n"
        "    \"mask_token_id\": 248070,\n"
        "    \"target_layer_ids\": [1, 16, 31, 46, 61]\n"
        "  }\n"
        "}\n";
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("qwen.json", json, path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "hidden_size") != NULL);
    ds4_dflash_config_free(&cfg);
}

static void test_real_deepseek_safetensors_layout_is_accepted(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;

    write_config("config.json", real_deepseek_dflash_json(), path, sizeof(path));
    write_safetensors_fixture(false);
    EXPECT(ds4_dflash_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_weights_validate(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(weights.loaded);
    EXPECT(weights.n_tensors == 62);
    EXPECT(strstr(weights.source_path, "model.safetensors") != NULL);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_safetensors_fc_shape_mismatch_is_rejected(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;

    write_config("config.json", real_deepseek_dflash_json(), path, sizeof(path));
    write_safetensors_fixture(true);
    EXPECT(ds4_dflash_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_weights_validate(&weights, temp_root, &cfg, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "fc.weight") != NULL);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_safetensors_fp8_dtype_is_recognized_before_reject(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;

    write_config("config.json", real_deepseek_dflash_json(), path, sizeof(path));
    write_safetensors_fixture_with_fc_dtype("F8_E4M3");
    EXPECT(ds4_dflash_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_weights_validate(&weights, temp_root, &cfg, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "F8_E4M3") != NULL);
    EXPECT(strstr(err, "BF16") != NULL);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_safetensors_open_reads_bf16_rows(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const ds4_dflash_tensor *embed = NULL;
    float row[4] = {0};

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(weights.loaded);
    EXPECT(weights.map != NULL);
    EXPECT(weights.n_bound_tensors == 18);
    embed = ds4_dflash_weights_find_tensor(&weights, "embed_tokens.weight");
    EXPECT(embed != NULL);
    EXPECT(embed && embed->shape[0] == 8 && embed->shape[1] == 4);
    EXPECT(ds4_dflash_tensor_read_bf16_f32(&weights, embed, 4, row, 4, err, sizeof(err)) == 0);
    EXPECT(row[0] == 1.0f);
    EXPECT(row[1] == 2.0f);
    EXPECT(row[2] == 3.0f);
    EXPECT(row[3] == 4.0f);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_dspark_sharded_mtp_artifact_is_bound_and_reads_bf16(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dspark_config cfg;
    ds4_dspark_weights weights;
    const ds4_dflash_tensor *main_proj = NULL;
    const ds4_dflash_tensor *main_norm = NULL;
    const ds4_dflash_tensor *hc_head_fn = NULL;
    float row[4] = {0};

    write_config("config.json", tiny_dspark_json(), path, sizeof(path));
    write_config("model.safetensors.index.json", tiny_dspark_index_json(), path, sizeof(path));
    write_tiny_dspark_shard0();
    write_tiny_dspark_shard2();

    ds4_dspark_config_init(&cfg);
    ds4_dspark_weights_init(&weights);
    EXPECT(ds4_dspark_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(cfg.loaded);
    EXPECT(cfg.block_size == 2);
    EXPECT(cfg.noise_token_id == 7);
    EXPECT(cfg.hidden_size == 4);
    EXPECT(cfg.vocab_size == 8);
    EXPECT(cfg.markov_rank == 3);
    EXPECT(cfg.sliding_window == 4);
    EXPECT(cfg.hc_mult == 2);
    EXPECT(cfg.hc_sinkhorn_iters == 20);
    EXPECT_NEAR(cfg.hc_eps, 1.0e-6f, 0.0000001f);
    EXPECT(cfg.n_target_layer_ids == 2);
    EXPECT(cfg.target_layer_ids[0] == 1);
    EXPECT(ds4_dspark_config_validate_target(&cfg, 4, 8, 4, err, sizeof(err)) == 0);

    EXPECT(ds4_dspark_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(weights.loaded);
    EXPECT(weights.n_shards == 2);
    EXPECT(weights.n_bound_tensors == 10);

    main_proj = ds4_dspark_weights_find_tensor(&weights, "mtp.0.main_proj.weight");
    EXPECT(main_proj != NULL);
    EXPECT(main_proj->dtype == DS4_DFLASH_TENSOR_F8_E4M3);
    EXPECT(main_proj->shape[0] == 4);
    EXPECT(main_proj->shape[1] == 8);
    EXPECT(main_proj->shard_index == 0);

    main_norm = ds4_dspark_weights_find_tensor(&weights, "mtp.0.main_norm.weight");
    EXPECT(main_norm != NULL);
    EXPECT(ds4_dspark_tensor_read_bf16_f32(&weights,
                                           main_norm,
                                           0,
                                           row,
                                           4,
                                           err,
                                           sizeof(err)) == 0);
    EXPECT_NEAR(row[0], 1.0f, 0.01f);
    EXPECT_NEAR(row[3], 4.0f, 0.01f);

    EXPECT(ds4_dspark_weights_read_bf16_f32(&weights,
                                            "mtp.2.norm.weight",
                                            0,
                                            row,
                                            4,
                                            err,
                                            sizeof(err)) == 0);
    EXPECT_NEAR(row[0], 10.0f, 0.01f);
    EXPECT_NEAR(row[3], 13.0f, 0.01f);

    hc_head_fn = ds4_dspark_weights_find_tensor(&weights, "mtp.2.hc_head_fn");
    EXPECT(hc_head_fn != NULL);
    EXPECT(hc_head_fn->dtype == DS4_DFLASH_TENSOR_F32);
    EXPECT(ds4_dspark_tensor_read_bf16_f32(&weights,
                                           hc_head_fn,
                                           0,
                                           row,
                                           1,
                                           err,
                                           sizeof(err)) != 0);
    EXPECT(strstr(err, "not BF16") != NULL);
    EXPECT(ds4_dspark_weights_read_f32(&weights,
                                       "mtp.2.hc_head_base",
                                       0,
                                       row,
                                       2,
                                       err,
                                       sizeof(err)) == 0);
    EXPECT_NEAR(row[0], 0.5f, 0.0001f);
    EXPECT_NEAR(row[1], 1.5f, 0.0001f);
    EXPECT(ds4_dspark_weights_read_f32(&weights,
                                       "mtp.2.hc_head_scale",
                                       0,
                                       row,
                                       1,
                                       err,
                                       sizeof(err)) == 0);
    EXPECT_NEAR(row[0], 2.0f, 0.0001f);

    ds4_dspark_weights_free(&weights);
    ds4_dspark_config_free(&cfg);
}

static void test_fp8_decode_helpers_match_known_values(void) {
    EXPECT_NEAR(ds4_dflash_fp8_e4m3_to_f32(0x38), 1.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp8_e4m3_to_f32(0xb8), -1.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp8_e4m3_to_f32(0x3c), 1.5f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp8_e4m3_to_f32(0x7e), 448.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp8_e8m0_to_f32(126), 0.5f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp8_e8m0_to_f32(127), 1.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp8_e8m0_to_f32(128), 2.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp4_e2m1_to_f32(0x1), 0.5f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp4_e2m1_to_f32(0x2), 1.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp4_e2m1_to_f32(0x7), 6.0f, 0.0001f);
    EXPECT_NEAR(ds4_dflash_fp4_e2m1_to_f32(0xa), -1.0f, 0.0001f);
}

static void test_dspark_hc_collapse_and_expand_match_reference_shape(void) {
    char err[256] = {0};
    unsigned char data[172] = {0};
    const uint64_t fn_off = 0;
    const uint64_t scale_off = 128;
    const uint64_t base_off = 140;
    ds4_dspark_weights weights;
    ds4_dflash_tensor fn = {
        .dtype = DS4_DFLASH_TENSOR_F32,
        .ndim = 2,
        .shape = {8, 4},
        .abs_offset = fn_off,
        .nbytes = 128,
        .shard_index = 0,
    };
    ds4_dflash_tensor scale = {
        .dtype = DS4_DFLASH_TENSOR_F32,
        .ndim = 1,
        .shape = {3},
        .abs_offset = scale_off,
        .nbytes = 12,
        .shard_index = 0,
    };
    ds4_dflash_tensor base = {
        .dtype = DS4_DFLASH_TENSOR_F32,
        .ndim = 1,
        .shape = {8},
        .abs_offset = base_off,
        .nbytes = 32,
        .shard_index = 0,
    };
    const float residual_hc[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    const float block_out[2] = {10.0f, 20.0f};
    float collapsed[2] = {0};
    float split[8] = {0};
    float out_hc[4] = {0};

    ds4_dspark_weights_init(&weights);
    weights.loaded = true;
    weights.n_shards = 1;
    weights.shards[0].map = data;
    weights.shards[0].file_size = sizeof(data);
    weights.shards[0].loaded = true;

    EXPECT(ds4_dspark_hc_collapse_f32(&weights,
                                      &fn,
                                      &scale,
                                      &base,
                                      residual_hc,
                                      2,
                                      2,
                                      3,
                                      1.0e-6f,
                                      collapsed,
                                      split,
                                      err,
                                      sizeof(err)) == 0);
    EXPECT_NEAR(split[0], 0.500001f, 0.00001f);
    EXPECT_NEAR(split[1], 0.500001f, 0.00001f);
    EXPECT_NEAR(split[2], 1.0f, 0.00001f);
    EXPECT_NEAR(split[3], 1.0f, 0.00001f);
    EXPECT_NEAR(split[4], 0.5f, 0.0001f);
    EXPECT_NEAR(split[7], 0.5f, 0.0001f);
    EXPECT_NEAR(collapsed[0], 4.000008f, 0.0001f);
    EXPECT_NEAR(collapsed[1], 6.000012f, 0.0001f);

    EXPECT(ds4_dspark_hc_expand_f32(block_out,
                                    residual_hc,
                                    split,
                                    2,
                                    2,
                                    out_hc,
                                    err,
                                    sizeof(err)) == 0);
    EXPECT_NEAR(out_hc[0], 14.0f, 0.001f);
    EXPECT_NEAR(out_hc[1], 26.0f, 0.001f);
    EXPECT_NEAR(out_hc[2], 14.0f, 0.001f);
    EXPECT_NEAR(out_hc[3], 26.0f, 0.001f);
}

static void test_dspark_unweighted_rms_norm_matches_q_norm_shape(void) {
    char err[256] = {0};
    const float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float rms = sqrtf(7.5f + 1.0e-6f);
    float out[4] = {0};

    EXPECT(ds4_dspark_rms_norm_f32(in, 4, out, err, sizeof(err)) == 0);
    EXPECT_NEAR(out[0], 1.0f / rms, 0.0001f);
    EXPECT_NEAR(out[1], 2.0f / rms, 0.0001f);
    EXPECT_NEAR(out[2], 3.0f / rms, 0.0001f);
    EXPECT_NEAR(out[3], 4.0f / rms, 0.0001f);
}

static void test_dspark_partial_rope_rotates_trailing_interleaved_pairs(void) {
    char err[256] = {0};
    float x[6] = {9.0f, 10.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const float c0 = cosf(1.0f);
    const float s0 = sinf(1.0f);
    const float c1 = cosf(0.01f);
    const float s1 = sinf(0.01f);

    EXPECT(ds4_dspark_apply_partial_rope_f32(x,
                                             1,
                                             6,
                                             4,
                                             1,
                                             10000.0f,
                                             0,
                                             1.0f,
                                             32.0f,
                                             1.0f,
                                             false,
                                             err,
                                             sizeof(err)) == 0);
    EXPECT_NEAR(x[0], 9.0f, 0.0001f);
    EXPECT_NEAR(x[1], 10.0f, 0.0001f);
    EXPECT_NEAR(x[2], c0, 0.0001f);
    EXPECT_NEAR(x[3], s0, 0.0001f);
    EXPECT_NEAR(x[4], -s1, 0.0001f);
    EXPECT_NEAR(x[5], c1, 0.0001f);
    EXPECT(ds4_dspark_apply_partial_rope_f32(x,
                                             1,
                                             6,
                                             4,
                                             1,
                                             10000.0f,
                                             0,
                                             1.0f,
                                             32.0f,
                                             1.0f,
                                             true,
                                             err,
                                             sizeof(err)) == 0);
    EXPECT_NEAR(x[2], 1.0f, 0.0001f);
    EXPECT_NEAR(x[3], 0.0f, 0.0001f);
    EXPECT_NEAR(x[4], 0.0f, 0.0001f);
    EXPECT_NEAR(x[5], 1.0f, 0.0001f);
}

static void test_dspark_sparse_attention_one_includes_sink(void) {
    char err[256] = {0};
    unsigned char data[8] = {0};
    ds4_dspark_weights weights;
    ds4_dflash_tensor sink = {
        .dtype = DS4_DFLASH_TENSOR_F32,
        .ndim = 1,
        .shape = {2},
        .abs_offset = 0,
        .nbytes = 8,
        .shard_index = 0,
    };
    const float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float kv[2] = {1.0f, 0.0f};
    float out[4] = {0};
    const float score0 = 1.0f / sqrtf(2.0f);
    const float weight0 = expf(score0) / (expf(score0) + 1.0f);

    ds4_dspark_weights_init(&weights);
    weights.loaded = true;
    weights.n_shards = 1;
    weights.shards[0].map = data;
    weights.shards[0].file_size = sizeof(data);
    weights.shards[0].loaded = true;

    EXPECT(ds4_dspark_sparse_attention_one_f32(&weights,
                                               &sink,
                                               q,
                                               kv,
                                               2,
                                               2,
                                               out,
                                               err,
                                               sizeof(err)) == 0);
    EXPECT_NEAR(out[0], weight0, 0.0001f);
    EXPECT_NEAR(out[1], 0.0f, 0.0001f);
    EXPECT_NEAR(out[2], 0.5f, 0.0001f);
    EXPECT_NEAR(out[3], 0.0f, 0.0001f);
}

static void test_dspark_sparse_attention_block_mixes_rows_and_sink(void) {
    char err[256] = {0};
    unsigned char data[4] = {0};
    ds4_dspark_weights weights;
    ds4_dflash_tensor sink = {
        .dtype = DS4_DFLASH_TENSOR_F32,
        .ndim = 1,
        .shape = {1},
        .abs_offset = 0,
        .nbytes = 4,
        .shard_index = 0,
    };
    const float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float kv[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float out[4] = {0};
    const float match = expf(1.0f / sqrtf(2.0f));
    const float denom = match + 2.0f;
    const float match_weight = match / denom;
    const float other_weight = 1.0f / denom;

    ds4_dspark_weights_init(&weights);
    weights.loaded = true;
    weights.n_shards = 1;
    weights.shards[0].map = data;
    weights.shards[0].file_size = sizeof(data);
    weights.shards[0].loaded = true;

    EXPECT(ds4_dspark_sparse_attention_block_f32(&weights,
                                                 &sink,
                                                 q,
                                                 kv,
                                                 2,
                                                 2,
                                                 1,
                                                 2,
                                                 out,
                                                 err,
                                                 sizeof(err)) == 0);
    EXPECT_NEAR(out[0], match_weight, 0.0001f);
    EXPECT_NEAR(out[1], other_weight, 0.0001f);
    EXPECT_NEAR(out[2], other_weight, 0.0001f);
    EXPECT_NEAR(out[3], match_weight, 0.0001f);
}

static void test_dspark_grouped_fp8_linear_uses_local_group_inputs(void) {
    char err[256] = {0};
    unsigned char data[9] = {0};
    ds4_dspark_weights weights;
    ds4_dflash_tensor weight = {
        .dtype = DS4_DFLASH_TENSOR_F8_E4M3,
        .ndim = 2,
        .shape = {4, 2},
        .abs_offset = 0,
        .nbytes = 8,
        .shard_index = 0,
    };
    ds4_dflash_tensor scale = {
        .dtype = DS4_DFLASH_TENSOR_F8_E8M0,
        .ndim = 2,
        .shape = {1, 1},
        .abs_offset = 8,
        .nbytes = 1,
        .shard_index = 0,
    };
    const float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0};

    data[0] = 0x38u;
    data[3] = 0x38u;
    data[4] = 0x38u;
    data[7] = 0x38u;
    data[8] = 127u;

    ds4_dspark_weights_init(&weights);
    weights.loaded = true;
    weights.n_shards = 1;
    weights.shards[0].map = data;
    weights.shards[0].file_size = sizeof(data);
    weights.shards[0].loaded = true;

    EXPECT(ds4_dspark_grouped_linear_f8_f32(&weights,
                                            &weight,
                                            &scale,
                                            in,
                                            2,
                                            out,
                                            err,
                                            sizeof(err)) == 0);
    EXPECT_NEAR(out[0], 1.0f, 0.0001f);
    EXPECT_NEAR(out[1], 2.0f, 0.0001f);
    EXPECT_NEAR(out[2], 3.0f, 0.0001f);
    EXPECT_NEAR(out[3], 4.0f, 0.0001f);
}

static void test_dspark_bf16_gate_topk_uses_bias_only_for_selection(void) {
    char err[256] = {0};
    unsigned char data[24] = {0};
    ds4_dspark_weights weights;
    ds4_dflash_tensor gate_weight = {
        .dtype = DS4_DFLASH_TENSOR_BF16,
        .ndim = 2,
        .shape = {3, 2},
        .abs_offset = 0,
        .nbytes = 12,
        .shard_index = 0,
    };
    ds4_dflash_tensor gate_bias = {
        .dtype = DS4_DFLASH_TENSOR_F32,
        .ndim = 1,
        .shape = {3},
        .abs_offset = 12,
        .nbytes = 12,
        .shard_index = 0,
    };
    const float in[2] = {1.0f, 1.0f};
    float logits[3] = {0};
    uint32_t indices[2] = {0};
    float route_weights[2] = {0};

    write_bf16(data, 0, 0, 1.0f);
    write_bf16(data, 0, 3, 2.0f);
    write_bf16(data, 0, 4, 1.0f);
    write_bf16(data, 0, 5, 1.0f);
    write_f32(data, 12, 2, 0.5f);

    ds4_dspark_weights_init(&weights);
    weights.loaded = true;
    weights.n_shards = 1;
    weights.shards[0].map = data;
    weights.shards[0].file_size = sizeof(data);
    weights.shards[0].loaded = true;

    EXPECT(ds4_dspark_linear_bf16_f32(&weights,
                                      &gate_weight,
                                      in,
                                      logits,
                                      err,
                                      sizeof(err)) == 0);
    EXPECT_NEAR(logits[0], 1.0f, 0.0001f);
    EXPECT_NEAR(logits[1], 2.0f, 0.0001f);
    EXPECT_NEAR(logits[2], 2.0f, 0.0001f);

    EXPECT(ds4_dspark_moe_gate_topk_f32(&weights,
                                        &gate_weight,
                                        &gate_bias,
                                        in,
                                        2,
                                        1.5f,
                                        indices,
                                        route_weights,
                                        err,
                                        sizeof(err)) == 0);
    EXPECT(indices[0] == 2);
    EXPECT(indices[1] == 1);
    EXPECT_NEAR(route_weights[0], 0.75f, 0.0001f);
    EXPECT_NEAR(route_weights[1], 0.75f, 0.0001f);
}

static void test_dspark_fp4_linear_decodes_packed_e2m1_weights(void) {
    char err[256] = {0};
    unsigned char data[6] = {0};
    ds4_dspark_weights weights;
    ds4_dflash_tensor weight = {
        .dtype = DS4_DFLASH_TENSOR_I8,
        .ndim = 2,
        .shape = {2, 2},
        .abs_offset = 0,
        .nbytes = 4,
        .shard_index = 0,
    };
    ds4_dflash_tensor scale = {
        .dtype = DS4_DFLASH_TENSOR_F8_E8M0,
        .ndim = 2,
        .shape = {2, 1},
        .abs_offset = 4,
        .nbytes = 2,
        .shard_index = 0,
    };
    const float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2] = {0};

    data[0] = 0x12u;
    data[1] = 0x4au;
    data[2] = 0x00u;
    data[3] = 0x00u;
    data[4] = 127u;
    data[5] = 128u;

    ds4_dspark_weights_init(&weights);
    weights.loaded = true;
    weights.n_shards = 1;
    weights.shards[0].map = data;
    weights.shards[0].file_size = sizeof(data);
    weights.shards[0].loaded = true;

    EXPECT(ds4_dspark_linear_fp4_f32(&weights,
                                     &weight,
                                     &scale,
                                     in,
                                     out,
                                     err,
                                     sizeof(err)) == 0);
    EXPECT_NEAR(out[0], 7.0f, 0.0001f);
    EXPECT_NEAR(out[1], 0.0f, 0.0001f);
}

static void test_dspark_swiglu_applies_deepseek_clamps(void) {
    char err[256] = {0};
    const float gate[3] = {20.0f, -2.0f, 1.0f};
    const float up[3] = {20.0f, -20.0f, 2.0f};
    float out[3] = {0};
    const float g0 = 10.0f;
    const float g1 = -2.0f;

    EXPECT(ds4_dspark_swiglu_f32(gate, up, 3, 10.0f, out, err, sizeof(err)) == 0);
    EXPECT_NEAR(out[0], (g0 / (1.0f + expf(-g0))) * 10.0f, 0.001f);
    EXPECT_NEAR(out[1], (g1 / (1.0f + expf(-g1))) * -10.0f, 0.001f);
    EXPECT_NEAR(out[2], (1.0f / (1.0f + expf(-1.0f))) * 2.0f, 0.001f);
}

static void test_dspark_main_projection_uses_fp8_weight_and_bf16_norm(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dspark_config cfg;
    ds4_dspark_weights weights;
    const ds4_dflash_tensor *main_proj = NULL;
    const ds4_dflash_tensor *main_scale = NULL;
    const ds4_dflash_tensor *main_norm = NULL;
    const float main_hidden[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float raw[4] = {0};
    float main_x[4] = {0};
    const float rms = sqrtf(7.5f + 1.0e-6f);

    write_config("config.json", tiny_dspark_json(), path, sizeof(path));
    write_config("model.safetensors.index.json", tiny_dspark_index_json(), path, sizeof(path));
    write_tiny_dspark_shard0();
    write_tiny_dspark_shard2();

    ds4_dspark_config_init(&cfg);
    ds4_dspark_weights_init(&weights);
    EXPECT(ds4_dspark_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(ds4_dspark_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    main_proj = ds4_dspark_weights_find_tensor(&weights, "mtp.0.main_proj.weight");
    main_scale = ds4_dspark_weights_find_tensor(&weights, "mtp.0.main_proj.scale");
    main_norm = ds4_dspark_weights_find_tensor(&weights, "mtp.0.main_norm.weight");
    EXPECT(ds4_dspark_linear_f8_f32(&weights,
                                    main_proj,
                                    main_scale,
                                    main_hidden,
                                    raw,
                                    err,
                                    sizeof(err)) == 0);
    EXPECT_NEAR(raw[0], 1.0f, 0.0001f);
    EXPECT_NEAR(raw[1], 2.0f, 0.0001f);
    EXPECT_NEAR(raw[2], 3.0f, 0.0001f);
    EXPECT_NEAR(raw[3], 4.0f, 0.0001f);
    EXPECT(ds4_dspark_rms_norm_bf16(&weights,
                                    main_norm,
                                    raw,
                                    4,
                                    main_x,
                                    err,
                                    sizeof(err)) == 0);
    EXPECT_NEAR(main_x[0], 1.0f / rms, 0.01f);
    EXPECT_NEAR(main_x[1], 4.0f / rms, 0.01f);
    EXPECT_NEAR(main_x[2], 9.0f / rms, 0.01f);
    EXPECT_NEAR(main_x[3], 16.0f / rms, 0.01f);
    memset(main_x, 0, sizeof(main_x));
    EXPECT(ds4_dspark_project_main_hidden(&weights,
                                          &cfg,
                                          main_hidden,
                                          main_x,
                                          err,
                                          sizeof(err)) == 0);
    EXPECT_NEAR(main_x[0], 1.0f / rms, 0.01f);
    EXPECT_NEAR(main_x[1], 4.0f / rms, 0.01f);
    EXPECT_NEAR(main_x[2], 9.0f / rms, 0.01f);
    EXPECT_NEAR(main_x[3], 16.0f / rms, 0.01f);

    ds4_dspark_weights_free(&weights);
    ds4_dspark_config_free(&cfg);
}

static void test_dspark_init_hc_block_from_main_repeats_rows_and_streams(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dspark_config cfg;
    const float main_x[4] = {1, 2, 3, 4};
    float hc[16];

    write_config("config.json", tiny_dspark_json(), path, sizeof(path));

    ds4_dspark_config_init(&cfg);
    EXPECT(ds4_dspark_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    for (uint32_t i = 0; i < 16u; i++) hc[i] = -1.0f;

    EXPECT(ds4_dspark_init_hc_block_from_main_f32(&cfg,
                                                  main_x,
                                                  2,
                                                  hc,
                                                  err,
                                                  sizeof(err)) == 0);
    for (uint32_t row = 0; row < 2u; row++) {
        for (uint32_t h = 0; h < 2u; h++) {
            for (uint32_t d = 0; d < 4u; d++) {
                EXPECT_NEAR(hc[(uint64_t)row * 8u + h * 4u + d], main_x[d], 0.0f);
            }
        }
    }
    EXPECT(ds4_dspark_init_hc_block_from_main_f32(&cfg,
                                                  main_x,
                                                  3,
                                                  hc,
                                                  err,
                                                  sizeof(err)) != 0);

    ds4_dspark_config_free(&cfg);
}

static void test_dspark_final_head_markov_and_confidence_reference(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dspark_config cfg;
    ds4_dspark_weights weights;
    const float hc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float hidden[4] = {0};
    float normed[4] = {0};
    float block_hc[16] = {0};
    float block_hidden[8] = {0};
    float block_normed[8] = {0};
    float markov_embedding[3] = {0};
    float markov_logits[8] = {0};
    float confidence_hidden[4] = {1, 1, 1, 1};
    float confidence = 0.0f;
    const float flat_rms = sqrtf(25.5f + 1.0e-6f);
    const float w0 = 1.0f / (1.0f + expf(-((hc[0] / flat_rms) * 2.0f + 0.5f))) + 1.0e-6f;
    const float w1 = 1.0f / (1.0f + expf(-((-hc[4] / flat_rms) * 2.0f + 1.5f))) + 1.0e-6f;
    const float h0 = w0 * hc[0] + w1 * hc[4];
    const float h1 = w0 * hc[1] + w1 * hc[5];
    const float h2 = w0 * hc[2] + w1 * hc[6];
    const float h3 = w0 * hc[3] + w1 * hc[7];
    const float hidden_rms = sqrtf((h0 * h0 + h1 * h1 + h2 * h2 + h3 * h3) / 4.0f + 1.0e-6f);

    write_config("config.json", tiny_dspark_json(), path, sizeof(path));
    write_config("model.safetensors.index.json", tiny_dspark_index_json(), path, sizeof(path));
    write_tiny_dspark_shard0();
    write_tiny_dspark_shard2();

    ds4_dspark_config_init(&cfg);
    ds4_dspark_weights_init(&weights);
    EXPECT(ds4_dspark_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(ds4_dspark_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);

    EXPECT(ds4_dspark_final_hc_head_f32(&weights,
                                        &cfg,
                                        hc,
                                        hidden,
                                        err,
                                        sizeof(err)) == 0);
    EXPECT_NEAR(hidden[0], h0, 0.0001f);
    EXPECT_NEAR(hidden[3], h3, 0.0001f);

    EXPECT(ds4_dspark_final_norm_f32(&weights,
                                     &cfg,
                                     hidden,
                                     normed,
                                     err,
                                     sizeof(err)) == 0);
    EXPECT_NEAR(normed[0], hidden[0] / hidden_rms * 10.0f, 0.01f);
    EXPECT_NEAR(normed[3], hidden[3] / hidden_rms * 13.0f, 0.01f);

    memcpy(block_hc, hc, sizeof(hc));
    memcpy(block_hc + 8, hc, sizeof(hc));
    EXPECT(ds4_dspark_final_block_norm_f32(&weights,
                                           &cfg,
                                           block_hc,
                                           2,
                                           block_hidden,
                                           block_normed,
                                           err,
                                           sizeof(err)) == 0);
    EXPECT_NEAR(block_hidden[0], hidden[0], 0.0001f);
    EXPECT_NEAR(block_hidden[3], hidden[3], 0.0001f);
    EXPECT_NEAR(block_hidden[4], hidden[0], 0.0001f);
    EXPECT_NEAR(block_hidden[7], hidden[3], 0.0001f);
    EXPECT_NEAR(block_normed[0], normed[0], 0.0001f);
    EXPECT_NEAR(block_normed[3], normed[3], 0.0001f);
    EXPECT_NEAR(block_normed[4], normed[0], 0.0001f);
    EXPECT_NEAR(block_normed[7], normed[3], 0.0001f);

    EXPECT(ds4_dspark_markov_prev_embedding_f32(&weights,
                                                &cfg,
                                                3,
                                                markov_embedding,
                                                err,
                                                sizeof(err)) == 0);
    EXPECT_NEAR(markov_embedding[0], 1.0f, 0.0001f);
    EXPECT_NEAR(markov_embedding[1], 2.0f, 0.0001f);
    EXPECT_NEAR(markov_embedding[2], 3.0f, 0.0001f);

    EXPECT(ds4_dspark_markov_logits_f32(&weights,
                                        &cfg,
                                        markov_embedding,
                                        markov_logits,
                                        err,
                                        sizeof(err)) == 0);
    EXPECT_NEAR(markov_logits[0], 1.0f, 0.0001f);
    EXPECT_NEAR(markov_logits[1], 4.0f, 0.0001f);
    EXPECT_NEAR(markov_logits[2], 9.0f, 0.0001f);
    EXPECT_NEAR(markov_logits[3], 6.0f, 0.0001f);

    EXPECT(ds4_dspark_confidence_logit_f32(&weights,
                                           &cfg,
                                           confidence_hidden,
                                           markov_embedding,
                                           &confidence,
                                           err,
                                           sizeof(err)) == 0);
    EXPECT_NEAR(confidence, 48.0f, 0.0001f);

    ds4_dspark_weights_free(&weights);
    ds4_dspark_config_free(&cfg);
}

static void test_dspark_select_draft_tokens_argmax_applies_markov_and_confidence(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dspark_config cfg;
    ds4_dspark_weights weights;
    float base_logits[16] = {0};
    const float hidden_rows[8] = {1, 1, 1, 1, 0, 0, 0, 0};
    uint32_t draft_tokens[2] = {0};
    float margins[2] = {0};
    float confidence_logits[2] = {0};
    int selected = 0;

    base_logits[8 + 4] = 10.0f;
    base_logits[8 + 5] = 3.0f;

    write_config("config.json", tiny_dspark_json(), path, sizeof(path));
    write_config("model.safetensors.index.json", tiny_dspark_index_json(), path, sizeof(path));
    write_tiny_dspark_shard0();
    write_tiny_dspark_shard2();

    ds4_dspark_config_init(&cfg);
    ds4_dspark_weights_init(&weights);
    EXPECT(ds4_dspark_config_load(&cfg, temp_root, err, sizeof(err)) == 0);
    EXPECT(ds4_dspark_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);

    selected = ds4_dspark_select_draft_tokens_argmax(&weights,
                                                     &cfg,
                                                     base_logits,
                                                     hidden_rows,
                                                     2,
                                                     3,
                                                     0.0f,
                                                     draft_tokens,
                                                     margins,
                                                     confidence_logits,
                                                     err,
                                                     sizeof(err));
    EXPECT(selected == 2);
    EXPECT(draft_tokens[0] == 2);
    EXPECT_NEAR(margins[0], 3.0f, 0.0001f);
    EXPECT_NEAR(confidence_logits[0], 48.0f, 0.0001f);
    EXPECT(draft_tokens[1] == 4);
    EXPECT_NEAR(margins[1], 7.0f, 0.0001f);
    EXPECT_NEAR(confidence_logits[1], 0.0f, 0.0001f);

    memset(draft_tokens, 0, sizeof(draft_tokens));
    memset(margins, 0, sizeof(margins));
    memset(confidence_logits, 0, sizeof(confidence_logits));
    selected = ds4_dspark_select_draft_tokens_argmax(&weights,
                                                     &cfg,
                                                     base_logits,
                                                     hidden_rows,
                                                     2,
                                                     3,
                                                     0.75f,
                                                     draft_tokens,
                                                     margins,
                                                     confidence_logits,
                                                     err,
                                                     sizeof(err));
    EXPECT(selected == 1);
    EXPECT(draft_tokens[0] == 2);
    EXPECT_NEAR(margins[0], 3.0f, 0.0001f);
    EXPECT_NEAR(confidence_logits[0], 48.0f, 0.0001f);

    memset(base_logits, 0, sizeof(base_logits));
    memset(draft_tokens, 0, sizeof(draft_tokens));
    memset(margins, 0, sizeof(margins));
    base_logits[6] = 5.0f;
    base_logits[1] = 2.0f;
    base_logits[8 + 5] = 4.0f;
    base_logits[8 + 4] = 1.5f;
    EXPECT(setenv("DS4_DSPARK_DISABLE_MARKOV", "1", 1) == 0);
    selected = ds4_dspark_select_draft_tokens_argmax(&weights,
                                                     &cfg,
                                                     base_logits,
                                                     NULL,
                                                     2,
                                                     3,
                                                     0.0f,
                                                     draft_tokens,
                                                     margins,
                                                     NULL,
                                                     err,
                                                     sizeof(err));
    EXPECT(unsetenv("DS4_DSPARK_DISABLE_MARKOV") == 0);
    EXPECT(selected == 2);
    EXPECT(draft_tokens[0] == 6);
    EXPECT_NEAR(margins[0], 3.0f, 0.0001f);
    EXPECT(draft_tokens[1] == 5);
    EXPECT_NEAR(margins[1], 2.5f, 0.0001f);

    ds4_dspark_weights_free(&weights);
    ds4_dspark_config_free(&cfg);
}

static void test_prepare_block_inputs_projects_taps(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    float taps[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float target_hidden[4] = {0};
    float noise[8] = {0};
    const float rms = sqrtf(7.5f + 1.0e-6f);

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_project_target_hidden(&weights,
                                            &cfg,
                                            taps,
                                            1,
                                            0,
                                            target_hidden,
                                            err,
                                            sizeof(err)) == 0);
    EXPECT_NEAR(target_hidden[0], 1.0f / rms, 0.01f);
    EXPECT_NEAR(target_hidden[1], 2.0f / rms, 0.01f);
    EXPECT_NEAR(target_hidden[2], 3.0f / rms, 0.01f);
    EXPECT_NEAR(target_hidden[3], 4.0f / rms, 0.01f);
    memset(noise, 0, sizeof(noise));
    EXPECT(ds4_dflash_prepare_noise_inputs(&weights,
                                           &cfg,
                                           1,
                                           noise,
                                           err,
                                           sizeof(err)) == 0);
    EXPECT_NEAR(noise[0], 1.0f, 0.02f);
    EXPECT_NEAR(noise[4], 1.0f, 0.02f);
    EXPECT(ds4_dflash_prepare_block_inputs(&weights,
                                           &cfg,
                                           taps,
                                           1,
                                           0,
                                           1,
                                           target_hidden,
                                           noise,
                                           err,
                                           sizeof(err)) == 0);
    EXPECT_NEAR(target_hidden[0], 1.0f / rms, 0.01f);
    EXPECT_NEAR(target_hidden[1], 2.0f / rms, 0.01f);
    EXPECT_NEAR(target_hidden[2], 3.0f / rms, 0.01f);
    EXPECT_NEAR(target_hidden[3], 4.0f / rms, 0.01f);
    EXPECT(noise[0] == 1.0f && noise[1] == 2.0f && noise[2] == 3.0f && noise[3] == 4.0f);
    EXPECT(noise[4] == 1.0f && noise[5] == 2.0f && noise[6] == 3.0f && noise[7] == 4.0f);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_eval_attention_uses_target_and_noise_kv(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float target_hidden[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float noise_hidden[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    const uint32_t target_pos[1] = {0};
    const uint32_t noise_pos[1] = {0};
    float out[4] = {0};
    const float noise_score = sqrtf(2.0f);
    const float noise_weight = expf(noise_score) / (1.0f + expf(noise_score));
    const float target_weight = 1.0f - noise_weight;

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_attention(&weights,
                                         &cfg,
                                         0,
                                         target_hidden,
                                         target_pos,
                                         1,
                                         noise_hidden,
                                         noise_pos,
                                         1,
                                         out,
                                         err,
                                         sizeof(err)) == 0);
    EXPECT_NEAR(out[0], target_weight, 0.02f);
    EXPECT_NEAR(out[1], 1.0f + 2.0f * noise_weight, 0.02f);
    EXPECT_NEAR(out[2], 0.5f, 0.02f);
    EXPECT_NEAR(out[3], 1.0f, 0.02f);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_eval_attention_honors_causal_sliding_block(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float noise_hidden[8] = {
        0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
    };
    const uint32_t noise_pos[2] = {0, 1};
    float causal[8] = {0};
    float noncausal[8] = {0};
    float row0_delta = 0.0f;
    float row1_delta = 0.0f;

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.sliding_window = 1;
    cfg.sliding_window_non_causal = false;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_attention(&weights,
                                         &cfg,
                                         0,
                                         NULL,
                                         NULL,
                                         0,
                                         noise_hidden,
                                         noise_pos,
                                         2,
                                         causal,
                                         err,
                                         sizeof(err)) == 0);
    cfg.sliding_window_non_causal = true;
    EXPECT(ds4_dflash_cpu_eval_attention(&weights,
                                         &cfg,
                                         0,
                                         NULL,
                                         NULL,
                                         0,
                                         noise_hidden,
                                         noise_pos,
                                         2,
                                         noncausal,
                                         err,
                                         sizeof(err)) == 0);
    for (int i = 0; i < 4; i++) {
        row0_delta += fabsf(causal[i] - noncausal[i]);
        row1_delta += fabsf(causal[4 + i] - noncausal[4 + i]);
    }
    EXPECT(row0_delta > 0.01f);
    EXPECT_NEAR(row1_delta, 0.0f, 0.02f);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_eval_mlp_uses_bound_bf16_weights(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float hidden[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, -1.0f, 2.0f, -2.0f,
    };
    float out[8] = {0};
    float expected[8] = {0};

    expected_tiny_mlp_row(hidden, expected);
    expected_tiny_mlp_row(hidden + 4, expected + 4);

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_mlp(&weights,
                                   &cfg,
                                   0,
                                   hidden,
                                   2,
                                   out,
                                   err,
                                   sizeof(err)) == 0);
    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(out[i], expected[i], 0.02f);
    }
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_eval_layer_and_block_compose_draft_graph(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float target_hidden[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float noise_hidden[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    const uint32_t target_pos[1] = {0};
    const uint32_t noise_pos[1] = {0};
    float attn[4] = {0};
    float expected[4] = {0};
    float layer_out[4] = {0};
    float block_out[4] = {0};

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_attention(&weights,
                                         &cfg,
                                         0,
                                         target_hidden,
                                         target_pos,
                                         1,
                                         noise_hidden,
                                         noise_pos,
                                         1,
                                         attn,
                                         err,
                                         sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_mlp(&weights,
                                   &cfg,
                                   0,
                                   attn,
                                   1,
                                   expected,
                                   err,
                                   sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_layer(&weights,
                                     &cfg,
                                     0,
                                     target_hidden,
                                     target_pos,
                                     1,
                                     noise_hidden,
                                     noise_pos,
                                     1,
                                     layer_out,
                                     err,
                                     sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_block(&weights,
                                     &cfg,
                                     target_hidden,
                                     target_pos,
                                     1,
                                     noise_hidden,
                                     noise_pos,
                                     1,
                                     block_out,
                                     err,
                                     sizeof(err)) == 0);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(layer_out[i], expected[i], 0.02f);
        EXPECT_NEAR(block_out[i], expected[i], 0.02f);
    }
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_eval_logits_selects_mapped_target_tokens(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float hidden[12] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 3.0f, 0.0f, 0.0f,
    };
    float logits[12] = {0};
    uint32_t draft_tokens[3] = {0};
    uint32_t target_tokens[3] = {0};
    uint32_t suffix_draft[2] = {0};
    uint32_t suffix_target[2] = {0};
    float suffix_margins[2] = {0};

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture();
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_logits(&weights,
                                      &cfg,
                                      hidden,
                                      3,
                                      logits,
                                      err,
                                      sizeof(err)) == 0);
    EXPECT_NEAR(logits[0], 0.0f, 0.02f);
    EXPECT_NEAR(logits[1], 1.0f, 0.02f);
    EXPECT_NEAR(logits[2], 2.0f, 0.02f);
    EXPECT_NEAR(logits[3], 3.0f, 0.02f);
    EXPECT_NEAR(logits[4], 0.0f, 0.02f);
    EXPECT_NEAR(logits[5], 2.0f, 0.02f);
    EXPECT_NEAR(logits[6], 0.0f, 0.02f);
    EXPECT_NEAR(logits[7], 0.0f, 0.02f);
    EXPECT_NEAR(logits[8], 0.0f, 0.02f);
    EXPECT_NEAR(logits[9], 0.0f, 0.02f);
    EXPECT_NEAR(logits[10], 4.0f, 0.02f);
    EXPECT_NEAR(logits[11], 0.0f, 0.02f);
    EXPECT(ds4_dflash_cpu_select_tokens(&weights,
                                        &cfg,
                                        logits,
                                        3,
                                        draft_tokens,
                                        target_tokens,
                                        err,
                                        sizeof(err)) == 0);
    EXPECT(draft_tokens[0] == 3 && target_tokens[0] == 7);
    EXPECT(draft_tokens[1] == 1 && target_tokens[1] == 3);
    EXPECT(draft_tokens[2] == 2 && target_tokens[2] == 5);
    EXPECT(ds4_dflash_cpu_select_draft_suffix_tokens(&weights,
                                                     &cfg,
                                                     logits,
                                                     3,
                                                     2,
                                                     suffix_draft,
                                                     suffix_target,
                                                     suffix_margins,
                                                     err,
                                                     sizeof(err)) == 0);
    EXPECT(suffix_draft[0] == 1 && suffix_target[0] == 3);
    EXPECT(suffix_draft[1] == 2 && suffix_target[1] == 5);
    EXPECT_NEAR(suffix_margins[0], 2.0f, 0.02f);
    EXPECT_NEAR(suffix_margins[1], 4.0f, 0.02f);
    EXPECT(ds4_dflash_cpu_select_draft_suffix_tokens(&weights,
                                                     &cfg,
                                                     logits,
                                                     3,
                                                     3,
                                                     suffix_draft,
                                                     suffix_target,
                                                     NULL,
                                                     err,
                                                     sizeof(err)) != 0);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_select_suffix_rejects_inadmissible_target_mapping(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float hidden[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    float logits[8] = {0};
    uint32_t draft_tokens[1] = {0};
    uint32_t target_tokens[1] = {0};

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture_with_mapping(true, false);
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_logits(&weights,
                                      &cfg,
                                      hidden,
                                      2,
                                      logits,
                                      err,
                                      sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_select_draft_suffix_tokens(&weights,
                                                     &cfg,
                                                     logits,
                                                     2,
                                                     1,
                                                     draft_tokens,
                                                     target_tokens,
                                                     NULL,
                                                     err,
                                                     sizeof(err)) != 0);
    EXPECT(strstr(err, "inadmissible") != NULL);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_cpu_select_suffix_rejects_out_of_vocab_target_mapping(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_weights weights;
    const float hidden[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    float logits[8] = {0};
    uint32_t draft_tokens[1] = {0};
    uint32_t target_tokens[1] = {0};

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.block_size = 2;
    cfg.mask_token_id = 1;
    cfg.hidden_size = 4;
    cfg.vocab_size = 8;
    cfg.draft_vocab_size = 4;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 3;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 2;
    cfg.hc_mult = 1;
    cfg.target_layer_ids[0] = 0;
    cfg.target_layer_ids[1] = 1;
    cfg.n_target_layer_ids = 2;

    write_tiny_safetensors_fixture_with_mapping(false, true);
    EXPECT(ds4_dflash_weights_open(&weights, temp_root, &cfg, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_eval_logits(&weights,
                                      &cfg,
                                      hidden,
                                      2,
                                      logits,
                                      err,
                                      sizeof(err)) == 0);
    EXPECT(ds4_dflash_cpu_select_draft_suffix_tokens(&weights,
                                                     &cfg,
                                                     logits,
                                                     2,
                                                     1,
                                                     draft_tokens,
                                                     target_tokens,
                                                     NULL,
                                                     err,
                                                     sizeof(err)) != 0);
    EXPECT(strstr(err, "outside target vocab") != NULL);
    ds4_dflash_weights_free(&weights);
    ds4_dflash_config_free(&cfg);
}

static void test_verify_stats_tracks_accept_and_reject_counts(void) {
    ds4_dflash_verify_stats stats;

    ds4_dflash_verify_stats_init(&stats, 3, 1);
    EXPECT(stats.drafted == 3);
    EXPECT(stats.verified == 0);
    EXPECT(stats.accepted_including_anchor == 1);
    EXPECT(stats.misses == 0);
    EXPECT(stats.rejected_draft_tokens == 0);
    EXPECT(stats.miss_index == -1);

    EXPECT(ds4_dflash_verify_step(&stats, 0, 11, 21, 21));
    EXPECT(stats.verified == 1);
    EXPECT(stats.accepted_including_anchor == 2);
    EXPECT(stats.misses == 0);
    EXPECT(stats.rejected_draft_tokens == 0);

    EXPECT(!ds4_dflash_verify_step(&stats, 1, 12, 22, 99));
    EXPECT(stats.verified == 1);
    EXPECT(stats.accepted_including_anchor == 2);
    EXPECT(stats.misses == 1);
    EXPECT(stats.rejected_draft_tokens == 2);
    EXPECT(stats.miss_index == 1);
    EXPECT(stats.miss_draft_token == 12);
    EXPECT(stats.miss_target_token == 22);
    EXPECT(stats.miss_target_top == 99);
    EXPECT(!ds4_dflash_verify_step(&stats, 2, 13, 23, 23));
    EXPECT(stats.verified == 1);
    EXPECT(stats.accepted_including_anchor == 2);
}

static void test_verify_stats_tracks_full_accept_counts(void) {
    ds4_dflash_verify_stats stats;

    ds4_dflash_verify_stats_init(&stats, 2, 1);
    EXPECT(ds4_dflash_verify_step(&stats, 0, 11, 21, 21));
    EXPECT(ds4_dflash_verify_step(&stats, 1, 12, 22, 22));
    EXPECT(stats.drafted == 2);
    EXPECT(stats.verified == 2);
    EXPECT(stats.accepted_including_anchor == 3);
    EXPECT(stats.misses == 0);
    EXPECT(stats.rejected_draft_tokens == 0);
    EXPECT(stats.miss_index == -1);
}

static void test_hidden_history_keeps_visible_prefix_rows(void) {
    char err[256] = {0};
    ds4_dflash_config cfg;
    ds4_dflash_hidden_history hist;
    const float row0[4] = {0, 1, 2, 3};
    const float row1[4] = {10, 11, 12, 13};
    const float row2[4] = {20, 21, 22, 23};
    const float row3[4] = {30, 31, 32, 33};
    float out_hidden[12] = {0};
    uint32_t out_pos[3] = {0};
    uint32_t n = 99;

    ds4_dflash_config_init(&cfg);
    cfg.loaded = true;
    cfg.hidden_size = 4;
    ds4_dflash_hidden_history_init(&hist);

    EXPECT(ds4_dflash_hidden_history_reserve(&hist, &cfg, 3, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_hidden_history_append(&hist, 0, row0, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_hidden_history_append(&hist, 1, row1, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_hidden_history_append(&hist, 2, row2, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_hidden_history_count_visible(&hist, 1, 0) == 1);
    EXPECT(ds4_dflash_hidden_history_count_visible(&hist, 2, 0) == 2);
    EXPECT(ds4_dflash_hidden_history_count_visible(&hist, 3, 0) == 3);
    EXPECT(ds4_dflash_hidden_history_copy_visible(&hist,
                                                  2,
                                                  0,
                                                  out_hidden,
                                                  out_pos,
                                                  &n,
                                                  err,
                                                  sizeof(err)) == 0);
    EXPECT(n == 2);
    EXPECT(out_pos[0] == 0 && out_pos[1] == 1);
    EXPECT_NEAR(out_hidden[0], 0.0f, 0.001f);
    EXPECT_NEAR(out_hidden[4], 10.0f, 0.001f);
    EXPECT(out_hidden[8] == 0.0f);

    EXPECT(ds4_dflash_hidden_history_append(&hist, 3, row3, err, sizeof(err)) == 0);
    memset(out_hidden, 0, sizeof(out_hidden));
    memset(out_pos, 0, sizeof(out_pos));
    EXPECT(ds4_dflash_hidden_history_copy_visible(&hist,
                                                  4,
                                                  0,
                                                  out_hidden,
                                                  out_pos,
                                                  &n,
                                                  err,
                                                  sizeof(err)) == 0);
    EXPECT(n == 3);
    EXPECT(out_pos[0] == 1 && out_pos[1] == 2 && out_pos[2] == 3);
    EXPECT_NEAR(out_hidden[0], 10.0f, 0.001f);
    EXPECT_NEAR(out_hidden[4], 20.0f, 0.001f);
    EXPECT_NEAR(out_hidden[8], 30.0f, 0.001f);

    memset(out_hidden, 0, sizeof(out_hidden));
    memset(out_pos, 0, sizeof(out_pos));
    EXPECT(ds4_dflash_hidden_history_copy_visible(&hist,
                                                  4,
                                                  2,
                                                  out_hidden,
                                                  out_pos,
                                                  &n,
                                                  err,
                                                  sizeof(err)) == 0);
    EXPECT(n == 2);
    EXPECT(out_pos[0] == 2 && out_pos[1] == 3);
    EXPECT_NEAR(out_hidden[0], 20.0f, 0.001f);
    EXPECT_NEAR(out_hidden[4], 30.0f, 0.001f);
    EXPECT(ds4_dflash_hidden_history_count_visible(&hist, 2, 0) == 1);
    EXPECT(ds4_dflash_hidden_history_append(&hist, 3, row3, err, sizeof(err)) != 0);
    ds4_dflash_hidden_history_rewind(&hist, 2);
    EXPECT(ds4_dflash_hidden_history_count_visible(&hist, 4, 0) == 1);
    EXPECT(ds4_dflash_hidden_history_append(&hist, 2, row2, err, sizeof(err)) == 0);

    ds4_dflash_hidden_history_reset(&hist);
    EXPECT(ds4_dflash_hidden_history_count_visible(&hist, 4, 0) == 0);
    EXPECT(ds4_dflash_hidden_history_append(&hist, 0, row0, err, sizeof(err)) == 0);

    ds4_dflash_hidden_history_free(&hist);
    ds4_dflash_config_free(&cfg);
}

static void test_target_layer_bounds_are_rejected(void) {
    const char *json =
        "{\n"
        "  \"block_size\": 16,\n"
        "  \"hidden_size\": 4096,\n"
        "  \"vocab_size\": 129280,\n"
        "  \"num_target_layers\": 43,\n"
        "  \"dflash_config\": {\n"
        "    \"mask_token_id\": 129279,\n"
        "    \"target_layer_ids\": [1, 43]\n"
        "  }\n"
        "}\n";
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("bad-layer.json", json, path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "target_layer_ids") != NULL);
    ds4_dflash_config_free(&cfg);
}

static void test_target_layers_must_be_strictly_increasing(void) {
    const char *json =
        "{\n"
        "  \"block_size\": 16,\n"
        "  \"hidden_size\": 4096,\n"
        "  \"vocab_size\": 129280,\n"
        "  \"num_target_layers\": 43,\n"
        "  \"dflash_config\": {\n"
        "    \"mask_token_id\": 129279,\n"
        "    \"target_layer_ids\": [3, 13, 12, 32, 42]\n"
        "  }\n"
        "}\n";
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("unsorted-layer.json", json, path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) == 0);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "strictly increasing") != NULL);
    ds4_dflash_config_free(&cfg);
}

static void test_aux_hidden_state_embedding_index_is_rejected(void) {
    const char *json =
        "{\n"
        "  \"block_size\": 8,\n"
        "  \"mask_token_id\": 1,\n"
        "  \"hidden_size\": 4096,\n"
        "  \"vocab_size\": 129280,\n"
        "  \"num_hidden_layers\": 5,\n"
        "  \"intermediate_size\": 2048,\n"
        "  \"num_attention_heads\": 64,\n"
        "  \"num_key_value_heads\": 1,\n"
        "  \"head_dim\": 256,\n"
        "  \"hc_mult\": 4,\n"
        "  \"aux_hidden_state_layer_ids\": [0, 12]\n"
        "}\n";
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("bad-aux-layer.json", json, path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "aux_hidden_state_layer_ids") != NULL);
}

static void test_missing_required_keys_are_rejected(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("missing.json",
                 "{ \"block_size\": 16, \"hidden_size\": 4096, \"vocab_size\": 129280, \"num_target_layers\": 43 }\n",
                 path,
                 sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) != 0);
    EXPECT(strstr(err, "mask_token_id") != NULL || strstr(err, "target_layer_ids") != NULL);
}

static void cleanup_temp_root(void) {
    char path[PATH_MAX];
    const char *names[] = {
        "valid.json",
        "config.json",
        "qwen.json",
        "bad-layer.json",
        "unsorted-layer.json",
        "bad-aux-layer.json",
        "missing.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "dspark-mtp-00001-of-00003.safetensors",
        "dspark-mtp-00003-of-00003.safetensors",
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        int n = snprintf(path, sizeof(path), "%s/%s", temp_root, names[i]);
        if (n >= 0 && (size_t)n < sizeof(path)) unlink(path);
    }
    rmdir(temp_root);
}

int main(void) {
    make_temp_root();
    test_valid_deepseek_config_file();
    test_directory_resolves_config_json();
    test_public_qwen_shape_is_rejected_for_deepseek();
    test_real_deepseek_safetensors_layout_is_accepted();
    test_safetensors_fc_shape_mismatch_is_rejected();
    test_safetensors_fp8_dtype_is_recognized_before_reject();
    test_safetensors_open_reads_bf16_rows();
    test_dspark_sharded_mtp_artifact_is_bound_and_reads_bf16();
    test_fp8_decode_helpers_match_known_values();
    test_dspark_hc_collapse_and_expand_match_reference_shape();
    test_dspark_unweighted_rms_norm_matches_q_norm_shape();
    test_dspark_partial_rope_rotates_trailing_interleaved_pairs();
    test_dspark_sparse_attention_one_includes_sink();
    test_dspark_sparse_attention_block_mixes_rows_and_sink();
    test_dspark_grouped_fp8_linear_uses_local_group_inputs();
    test_dspark_bf16_gate_topk_uses_bias_only_for_selection();
    test_dspark_fp4_linear_decodes_packed_e2m1_weights();
    test_dspark_swiglu_applies_deepseek_clamps();
    test_dspark_main_projection_uses_fp8_weight_and_bf16_norm();
    test_dspark_init_hc_block_from_main_repeats_rows_and_streams();
    test_dspark_final_head_markov_and_confidence_reference();
    test_dspark_select_draft_tokens_argmax_applies_markov_and_confidence();
    test_prepare_block_inputs_projects_taps();
    test_cpu_eval_attention_uses_target_and_noise_kv();
    test_cpu_eval_attention_honors_causal_sliding_block();
    test_cpu_eval_mlp_uses_bound_bf16_weights();
    test_cpu_eval_layer_and_block_compose_draft_graph();
    test_cpu_eval_logits_selects_mapped_target_tokens();
    test_cpu_select_suffix_rejects_inadmissible_target_mapping();
    test_cpu_select_suffix_rejects_out_of_vocab_target_mapping();
    test_verify_stats_tracks_accept_and_reject_counts();
    test_verify_stats_tracks_full_accept_counts();
    test_hidden_history_keeps_visible_prefix_rows();
    test_target_layer_bounds_are_rejected();
    test_target_layers_must_be_strictly_increasing();
    test_aux_hidden_state_embedding_index_is_rejected();
    test_missing_required_keys_are_rejected();
    cleanup_temp_root();

    if (failures) {
        fprintf(stderr, "ds4_dflash_config_test: %d failure(s)\n", failures);
        return 1;
    }
    printf("ds4_dflash_config_test: OK\n");
    return 0;
}
