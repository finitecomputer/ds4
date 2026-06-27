#include "ds4_dflash.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int dflash_err(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
        err[errlen - 1] = '\0';
    }
    return 1;
}

void ds4_dflash_config_init(ds4_dflash_config *cfg) {
    if (cfg) memset(cfg, 0, sizeof(*cfg));
}

void ds4_dflash_config_free(ds4_dflash_config *cfg) {
    ds4_dflash_config_init(cfg);
}

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static const char *json_key_value(const char *json, const char *key) {
    const size_t key_len = strlen(key);
    const char *p = json;
    while ((p = strchr(p, '"')) != NULL) {
        const char *start = p + 1;
        const char *q = start;
        while (*q) {
            if (*q == '\\' && q[1]) {
                q += 2;
                continue;
            }
            if (*q == '"') break;
            q++;
        }
        if (!*q) return NULL;
        if ((size_t)(q - start) == key_len && memcmp(start, key, key_len) == 0) {
            const char *colon = skip_ws(q + 1);
            if (*colon == ':') return colon + 1;
        }
        p = q + 1;
    }
    return NULL;
}

static int parse_u32_at(const char *p,
                        uint32_t *out,
                        const char **endp,
                        const char *name,
                        char *err,
                        size_t errlen) {
    char *end = NULL;
    unsigned long long v = 0;

    p = skip_ws(p);
    if (!isdigit((unsigned char)*p)) {
        return dflash_err(err, errlen, "DFlash config key '%s' must be an unsigned integer", name);
    }
    errno = 0;
    v = strtoull(p, &end, 10);
    if (end == p || errno == ERANGE || v > UINT32_MAX) {
        return dflash_err(err, errlen, "DFlash config key '%s' is outside uint32 range", name);
    }
    *out = (uint32_t)v;
    if (endp) *endp = end;
    return 0;
}

static int parse_u32_key(const char *json,
                         const char *key,
                         uint32_t *out,
                         bool required,
                         char *err,
                         size_t errlen) {
    const char *p = json_key_value(json, key);
    if (!p) {
        if (!required) return 0;
        return dflash_err(err, errlen, "DFlash config is missing required key '%s'", key);
    }
    return parse_u32_at(p, out, NULL, key, err, errlen);
}

static int parse_u32_array_key(const char *json,
                               const char *key,
                               uint32_t *out,
                               uint32_t cap,
                               uint32_t *len,
                               char *err,
                               size_t errlen) {
    const char *p = json_key_value(json, key);
    uint32_t n = 0;

    if (!p) return dflash_err(err, errlen, "DFlash config is missing required key '%s'", key);
    p = skip_ws(p);
    if (*p != '[') {
        return dflash_err(err, errlen, "DFlash config key '%s' must be an array", key);
    }
    p++;
    for (;;) {
        uint32_t v = 0;
        p = skip_ws(p);
        if (*p == ']') {
            p++;
            break;
        }
        if (n >= cap) {
            return dflash_err(err, errlen,
                              "DFlash config key '%s' has more than %u entries",
                              key, cap);
        }
        if (parse_u32_at(p, &v, &p, key, err, errlen) != 0) return 1;
        out[n++] = v;
        p = skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == ']') {
            p++;
            break;
        }
        return dflash_err(err, errlen, "DFlash config key '%s' has invalid array syntax", key);
    }
    *len = n;
    return 0;
}

static int resolve_config_path(const char *path,
                               char *out,
                               size_t outlen,
                               char *err,
                               size_t errlen) {
    struct stat st;
    int n = 0;

    if (!path || !path[0]) return dflash_err(err, errlen, "DFlash config path is empty");
    if (stat(path, &st) != 0) {
        return dflash_err(err, errlen, "DFlash config path '%s': %s", path, strerror(errno));
    }
    if (S_ISDIR(st.st_mode)) {
        n = snprintf(out, outlen, "%s/config.json", path);
        if (n < 0 || (size_t)n >= outlen) {
            return dflash_err(err, errlen, "DFlash config path is too long");
        }
        if (stat(out, &st) != 0) {
            return dflash_err(err, errlen, "DFlash config file '%s': %s", out, strerror(errno));
        }
        if (!S_ISREG(st.st_mode)) {
            return dflash_err(err, errlen, "DFlash config file '%s' is not a regular file", out);
        }
        return 0;
    }
    if (!S_ISREG(st.st_mode)) {
        return dflash_err(err, errlen, "DFlash config path '%s' is not a regular file", path);
    }
    n = snprintf(out, outlen, "%s", path);
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "DFlash config path is too long");
    }
    return 0;
}

static char *read_file(const char *path, char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    long len = 0;
    char *buf = NULL;
    size_t got = 0;

    if (!fp) {
        dflash_err(err, errlen, "DFlash config file '%s': %s", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        dflash_err(err, errlen, "DFlash config file '%s': seek failed", path);
        fclose(fp);
        return NULL;
    }
    len = ftell(fp);
    if (len < 0) {
        dflash_err(err, errlen, "DFlash config file '%s': size failed", path);
        fclose(fp);
        return NULL;
    }
    if ((unsigned long)len > (unsigned long)SIZE_MAX - 1ul) {
        dflash_err(err, errlen, "DFlash config file '%s' is too large", path);
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    buf = malloc((size_t)len + 1u);
    if (!buf) {
        dflash_err(err, errlen, "out of memory reading DFlash config");
        fclose(fp);
        return NULL;
    }
    got = fread(buf, 1, (size_t)len, fp);
    if (got != (size_t)len) {
        dflash_err(err, errlen, "DFlash config file '%s': read failed", path);
        free(buf);
        fclose(fp);
        return NULL;
    }
    buf[len] = '\0';
    fclose(fp);
    return buf;
}

int ds4_dflash_config_load(ds4_dflash_config *cfg,
                           const char *path,
                           char *err,
                           size_t errlen) {
    char resolved[DS4_DFLASH_MAX_PATH];
    char *json = NULL;

    if (!cfg) return dflash_err(err, errlen, "DFlash config output is null");
    ds4_dflash_config_init(cfg);
    if (resolve_config_path(path, resolved, sizeof(resolved), err, errlen) != 0) return 1;
    json = read_file(resolved, err, errlen);
    if (!json) return 1;

    if (parse_u32_key(json, "block_size", &cfg->block_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "mask_token_id", &cfg->mask_token_id, true, err, errlen) != 0 ||
        parse_u32_key(json, "hidden_size", &cfg->hidden_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "vocab_size", &cfg->vocab_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "num_target_layers", &cfg->num_target_layers, true, err, errlen) != 0 ||
        parse_u32_key(json, "num_hidden_layers", &cfg->num_hidden_layers, false, err, errlen) != 0 ||
        parse_u32_array_key(json,
                            "target_layer_ids",
                            cfg->target_layer_ids,
                            DS4_DFLASH_MAX_TARGET_LAYERS,
                            &cfg->n_target_layer_ids,
                            err,
                            errlen) != 0) {
        free(json);
        ds4_dflash_config_init(cfg);
        return 1;
    }

    snprintf(cfg->source_path, sizeof(cfg->source_path), "%s", resolved);
    cfg->loaded = true;
    free(json);
    return 0;
}

int ds4_dflash_config_validate_target(const ds4_dflash_config *cfg,
                                      uint32_t target_hidden_size,
                                      uint32_t target_vocab_size,
                                      uint32_t target_n_layer,
                                      char *err,
                                      size_t errlen) {
    if (!cfg || !cfg->loaded) {
        return dflash_err(err, errlen, "DFlash config is not loaded");
    }
    if (cfg->block_size < 2 || cfg->block_size > 64) {
        return dflash_err(err, errlen,
                          "DFlash block_size %u is outside the supported range 2..64",
                          cfg->block_size);
    }
    if (cfg->hidden_size != target_hidden_size) {
        return dflash_err(err, errlen,
                          "DFlash hidden_size %u does not match target hidden_size %u",
                          cfg->hidden_size, target_hidden_size);
    }
    if (cfg->vocab_size != target_vocab_size) {
        return dflash_err(err, errlen,
                          "DFlash vocab_size %u does not match target vocab_size %u",
                          cfg->vocab_size, target_vocab_size);
    }
    if (cfg->num_target_layers != target_n_layer) {
        return dflash_err(err, errlen,
                          "DFlash num_target_layers %u does not match target layer count %u",
                          cfg->num_target_layers, target_n_layer);
    }
    if (cfg->mask_token_id >= cfg->vocab_size) {
        return dflash_err(err, errlen,
                          "DFlash mask_token_id %u is outside vocab_size %u",
                          cfg->mask_token_id, cfg->vocab_size);
    }
    if (cfg->n_target_layer_ids == 0) {
        return dflash_err(err, errlen, "DFlash target_layer_ids must not be empty");
    }
    for (uint32_t i = 0; i < cfg->n_target_layer_ids; i++) {
        if (cfg->target_layer_ids[i] >= target_n_layer) {
            return dflash_err(err, errlen,
                              "DFlash target_layer_ids[%u]=%u is outside target layer count %u",
                              i, cfg->target_layer_ids[i], target_n_layer);
        }
        for (uint32_t j = 0; j < i; j++) {
            if (cfg->target_layer_ids[j] == cfg->target_layer_ids[i]) {
                return dflash_err(err, errlen,
                                  "DFlash target_layer_ids contains duplicate layer %u",
                                  cfg->target_layer_ids[i]);
            }
        }
    }
    return 0;
}
