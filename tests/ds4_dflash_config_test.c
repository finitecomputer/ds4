#include "ds4_dflash.h"

#include <errno.h>
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

static const char *deepseek_shape_json(void) {
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

static void test_valid_deepseek_config_file(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("valid.json", deepseek_shape_json(), path, sizeof(path));
    EXPECT(ds4_dflash_config_load(&cfg, path, err, sizeof(err)) == 0);
    EXPECT(cfg.loaded);
    EXPECT(cfg.block_size == 16);
    EXPECT(cfg.hidden_size == 4096);
    EXPECT(cfg.vocab_size == 129280);
    EXPECT(cfg.num_target_layers == 43);
    EXPECT(cfg.n_target_layer_ids == 5);
    EXPECT(ds4_dflash_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) == 0);
    ds4_dflash_config_free(&cfg);
}

static void test_directory_resolves_config_json(void) {
    char path[PATH_MAX];
    char err[256] = {0};
    ds4_dflash_config cfg;

    write_config("config.json", deepseek_shape_json(), path, sizeof(path));
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
