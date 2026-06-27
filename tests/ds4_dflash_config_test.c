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

static void write_tiny_safetensors_fixture(void) {
    char path[PATH_MAX];
    char *header = calloc(1, 65536);
    test_buf b = {.ptr = header, .cap = 65536};
    bool first = true;
    uint64_t off = 0;
    uint64_t embed_off = 0;
    uint64_t fc_off = 0;
    uint64_t hidden_norm_off = 0;
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
    append_tiny_tensor1(&b, &first, "d2t", "I64", 4, &off);
    append_tiny_tensor1(&b, &first, "t2d", "BOOL", 8, &off);
    append_tiny_tensor2(&b, &first, "embed_tokens.weight", "BF16", 8, 4, &off, &embed_off);
    append_tiny_tensor2(&b, &first, "fc.weight", "BF16", 4, 8, &off, &fc_off);
    hidden_norm_off = off;
    append_tiny_tensor1(&b, &first, "hidden_norm.weight", "BF16", 4, &off);
    append_tiny_tensor1(&b, &first, "norm.weight", "BF16", 4, &off);
    append_tiny_tensor2(&b, &first, "lm_head.weight", "BF16", 4, 4, &off, NULL);
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
    for (int i = 0; i < 4; i++) {
        write_bf16(data, embed_off, (uint64_t)(4 + i), (float)(i + 1));
        write_bf16(data, fc_off, (uint64_t)(i * 8 + i), 1.0f);
        write_bf16(data, hidden_norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, input_norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, post_norm_off, (uint64_t)i, 1.0f);
        write_bf16(data, q_proj_off, (uint64_t)(i * 4 + i), 1.0f);
        write_bf16(data, o_proj_off, (uint64_t)(i * 4 + i), 1.0f);
    }
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

static void test_valid_deepseek_config_file(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("valid.json", real_deepseek_dflash_json(), path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) == 0);
    EXPECT(cfg.loaded);
    EXPECT(cfg.block_size == 8);
    EXPECT(cfg.hidden_size == 4096);
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
    EXPECT(cfg.n_target_layer_ids == 5);
    EXPECT(cfg.target_layer_ids[0] == 3);
    EXPECT(cfg.target_layer_ids[4] == 42);
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
    test_cpu_eval_mlp_uses_bound_bf16_weights();
    test_target_layer_bounds_are_rejected();
    test_missing_required_keys_are_rejected();
    cleanup_temp_root();

    if (failures) {
        fprintf(stderr, "ds4_dflash_config_test: %d failure(s)\n", failures);
        return 1;
    }
    printf("ds4_dflash_config_test: OK\n");
    return 0;
}
