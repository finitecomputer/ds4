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
           "  }\n"
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
    EXPECT(cfg.sliding_window == 2048);
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
                                                     err,
                                                     sizeof(err)) == 0);
    EXPECT(suffix_draft[0] == 1 && suffix_target[0] == 3);
    EXPECT(suffix_draft[1] == 2 && suffix_target[1] == 5);
    EXPECT(ds4_dflash_cpu_select_draft_suffix_tokens(&weights,
                                                     &cfg,
                                                     logits,
                                                     3,
                                                     3,
                                                     suffix_draft,
                                                     suffix_target,
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
    test_safetensors_open_reads_bf16_rows();
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
