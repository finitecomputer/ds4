#include "ds4_dflash.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
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
