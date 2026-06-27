#ifndef DS4_DFLASH_H
#define DS4_DFLASH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_DFLASH_MAX_TARGET_LAYERS 128u
#define DS4_DFLASH_MAX_PATH 4096u

typedef struct {
    char source_path[DS4_DFLASH_MAX_PATH];
    uint32_t block_size;
    uint32_t mask_token_id;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t num_target_layers;
    uint32_t num_hidden_layers;
    uint32_t target_layer_ids[DS4_DFLASH_MAX_TARGET_LAYERS];
    uint32_t n_target_layer_ids;
    bool loaded;
} ds4_dflash_config;

void ds4_dflash_config_init(ds4_dflash_config *cfg);
void ds4_dflash_config_free(ds4_dflash_config *cfg);

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

#endif
