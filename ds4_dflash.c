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

void ds4_dflash_weights_init(ds4_dflash_weights *w) {
    if (w) memset(w, 0, sizeof(*w));
}

void ds4_dflash_weights_free(ds4_dflash_weights *w) {
    ds4_dflash_weights_init(w);
}

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static const char *json_key_value_bounded(const char *json, const char *end, const char *key) {
    const size_t key_len = strlen(key);
    const char *p = json;
    while ((!end || p < end) && (p = strchr(p, '"')) != NULL) {
        if (end && p >= end) return NULL;
        const char *start = p + 1;
        const char *q = start;
        while (*q && (!end || q < end)) {
            if (*q == '\\' && q[1]) {
                q += 2;
                continue;
            }
            if (*q == '"') break;
            q++;
        }
        if (!*q || (end && q >= end)) return NULL;
        if ((size_t)(q - start) == key_len && memcmp(start, key, key_len) == 0) {
            const char *colon = skip_ws(q + 1);
            if (end && colon >= end) return NULL;
            if (*colon == ':') return colon + 1;
        }
        p = q + 1;
    }
    return NULL;
}

static const char *json_key_value(const char *json, const char *key) {
    return json_key_value_bounded(json, NULL, key);
}

static int json_object_span(const char *value, const char **obj_start, const char **obj_end) {
    const char *p = skip_ws(value);
    uint32_t depth = 0;
    bool in_string = false;

    if (*p != '{') return 1;
    *obj_start = p;
    for (; *p; p++) {
        if (in_string) {
            if (*p == '\\' && p[1]) {
                p++;
                continue;
            }
            if (*p == '"') in_string = false;
            continue;
        }
        if (*p == '"') {
            in_string = true;
        } else if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            if (depth == 0) return 1;
            depth--;
            if (depth == 0) {
                *obj_end = p + 1;
                return 0;
            }
        }
    }
    return 1;
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

static int parse_u32_array_key_optional(const char *json,
                                        const char *key,
                                        uint32_t *out,
                                        uint32_t cap,
                                        uint32_t *len,
                                        bool *found,
                                        char *err,
                                        size_t errlen) {
    if (!json_key_value(json, key)) {
        if (found) *found = false;
        return 0;
    }
    if (found) *found = true;
    return parse_u32_array_key(json, key, out, cap, len, err, errlen);
}

static int parse_u64_array_at(const char *p,
                              uint64_t *out,
                              uint32_t cap,
                              uint32_t *len,
                              const char *name,
                              char *err,
                              size_t errlen) {
    uint32_t n = 0;

    p = skip_ws(p);
    if (*p != '[') {
        return dflash_err(err, errlen, "DFlash safetensors key '%s' must be an array", name);
    }
    p++;
    for (;;) {
        char *end = NULL;
        unsigned long long v = 0;

        p = skip_ws(p);
        if (*p == ']') {
            p++;
            break;
        }
        if (n >= cap) {
            return dflash_err(err, errlen,
                              "DFlash safetensors key '%s' has more than %u entries",
                              name, cap);
        }
        if (!isdigit((unsigned char)*p)) {
            return dflash_err(err, errlen,
                              "DFlash safetensors key '%s' must contain unsigned integers",
                              name);
        }
        errno = 0;
        v = strtoull(p, &end, 10);
        if (end == p || errno == ERANGE) {
            return dflash_err(err, errlen,
                              "DFlash safetensors key '%s' has an out-of-range integer",
                              name);
        }
        out[n++] = (uint64_t)v;
        p = skip_ws(end);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == ']') {
            p++;
            break;
        }
        return dflash_err(err, errlen,
                          "DFlash safetensors key '%s' has invalid array syntax",
                          name);
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

static bool str_ends_with(const char *s, const char *suffix) {
    const size_t n = s ? strlen(s) : 0;
    const size_t m = suffix ? strlen(suffix) : 0;
    return n >= m && memcmp(s + n - m, suffix, m) == 0;
}

static int resolve_weights_path(const char *path,
                                char *out,
                                size_t outlen,
                                char *err,
                                size_t errlen) {
    struct stat st;
    int n = 0;

    if (!path || !path[0]) return dflash_err(err, errlen, "DFlash weights path is empty");
    if (stat(path, &st) != 0) {
        return dflash_err(err, errlen, "DFlash weights path '%s': %s", path, strerror(errno));
    }
    if (S_ISDIR(st.st_mode)) {
        n = snprintf(out, outlen, "%s/model.safetensors", path);
    } else if (S_ISREG(st.st_mode) && str_ends_with(path, ".safetensors")) {
        n = snprintf(out, outlen, "%s", path);
    } else if (S_ISREG(st.st_mode)) {
        const char *slash = strrchr(path, '/');
        if (slash) {
            n = snprintf(out, outlen, "%.*s/model.safetensors",
                         (int)(slash - path), path);
        } else {
            n = snprintf(out, outlen, "model.safetensors");
        }
    } else {
        return dflash_err(err, errlen, "DFlash weights path '%s' is not a regular file or directory", path);
    }
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "DFlash weights path is too long");
    }
    if (stat(out, &st) != 0) {
        return dflash_err(err, errlen, "DFlash weights file '%s': %s", out, strerror(errno));
    }
    if (!S_ISREG(st.st_mode)) {
        return dflash_err(err, errlen, "DFlash weights file '%s' is not a regular file", out);
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
    bool have_aux_layer_ids = false;
    bool have_legacy_layer_ids = false;

    if (!cfg) return dflash_err(err, errlen, "DFlash config output is null");
    ds4_dflash_config_init(cfg);
    if (resolve_config_path(path, resolved, sizeof(resolved), err, errlen) != 0) return 1;
    json = read_file(resolved, err, errlen);
    if (!json) return 1;

    if (parse_u32_key(json, "block_size", &cfg->block_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "mask_token_id", &cfg->mask_token_id, true, err, errlen) != 0 ||
        parse_u32_key(json, "hidden_size", &cfg->hidden_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "vocab_size", &cfg->vocab_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "draft_vocab_size", &cfg->draft_vocab_size, false, err, errlen) != 0 ||
        parse_u32_key(json, "target_hidden_size", &cfg->target_hidden_size, false, err, errlen) != 0 ||
        parse_u32_key(json, "num_target_layers", &cfg->num_target_layers, false, err, errlen) != 0 ||
        parse_u32_key(json, "num_hidden_layers", &cfg->num_hidden_layers, false, err, errlen) != 0 ||
        parse_u32_key(json, "intermediate_size", &cfg->intermediate_size, false, err, errlen) != 0 ||
        parse_u32_key(json, "num_attention_heads", &cfg->num_attention_heads, false, err, errlen) != 0 ||
        parse_u32_key(json, "num_key_value_heads", &cfg->num_key_value_heads, false, err, errlen) != 0 ||
        parse_u32_key(json, "head_dim", &cfg->head_dim, false, err, errlen) != 0 ||
        parse_u32_key(json, "hc_mult", &cfg->hc_mult, false, err, errlen) != 0 ||
        parse_u32_key(json, "sliding_window", &cfg->sliding_window, false, err, errlen) != 0 ||
        parse_u32_key(json, "max_anchors", &cfg->max_anchors, false, err, errlen) != 0 ||
        parse_u32_array_key_optional(json,
                                     "aux_hidden_state_layer_ids",
                                     cfg->target_layer_ids,
                                     DS4_DFLASH_MAX_TARGET_LAYERS,
                                     &cfg->n_target_layer_ids,
                                     &have_aux_layer_ids,
                                     err,
                                     errlen) != 0) {
        free(json);
        ds4_dflash_config_init(cfg);
        return 1;
    }
    if (!have_aux_layer_ids &&
        parse_u32_array_key_optional(json,
                                     "target_layer_ids",
                                     cfg->target_layer_ids,
                                     DS4_DFLASH_MAX_TARGET_LAYERS,
                                     &cfg->n_target_layer_ids,
                                     &have_legacy_layer_ids,
                                     err,
                                     errlen) != 0) {
        free(json);
        ds4_dflash_config_init(cfg);
        return 1;
    }
    if (!have_aux_layer_ids && !have_legacy_layer_ids) {
        free(json);
        ds4_dflash_config_init(cfg);
        return dflash_err(err, errlen,
                          "DFlash config is missing required key 'aux_hidden_state_layer_ids'");
    }
    if (cfg->draft_vocab_size == 0) cfg->draft_vocab_size = cfg->vocab_size;
    if (cfg->target_hidden_size == 0) cfg->target_hidden_size = cfg->hidden_size;

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
    if (cfg->target_hidden_size != 0 && cfg->target_hidden_size != target_hidden_size) {
        return dflash_err(err, errlen,
                          "DFlash target_hidden_size %u does not match target hidden_size %u",
                          cfg->target_hidden_size, target_hidden_size);
    }
    if (cfg->vocab_size != target_vocab_size) {
        return dflash_err(err, errlen,
                          "DFlash vocab_size %u does not match target vocab_size %u",
                          cfg->vocab_size, target_vocab_size);
    }
    if (cfg->draft_vocab_size == 0 || cfg->draft_vocab_size > cfg->vocab_size) {
        return dflash_err(err, errlen,
                          "DFlash draft_vocab_size %u is invalid for target vocab_size %u",
                          cfg->draft_vocab_size, cfg->vocab_size);
    }
    if (cfg->num_target_layers != 0 && cfg->num_target_layers != target_n_layer) {
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

typedef enum {
    DFLASH_ST_DTYPE_UNKNOWN = 0,
    DFLASH_ST_DTYPE_BF16,
    DFLASH_ST_DTYPE_BOOL,
    DFLASH_ST_DTYPE_I64,
} dflash_st_dtype;

typedef struct {
    dflash_st_dtype dtype;
    uint64_t shape[4];
    uint32_t ndim;
} dflash_tensor_meta;

static uint64_t read_le64(const unsigned char b[8]) {
    return ((uint64_t)b[0]) |
           ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) |
           ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) |
           ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
}

static char *read_safetensors_header(const char *path,
                                     uint64_t *header_len,
                                     char *err,
                                     size_t errlen) {
    FILE *fp = fopen(path, "rb");
    unsigned char len_bytes[8];
    uint64_t len = 0;
    char *header = NULL;

    if (!fp) {
        dflash_err(err, errlen, "DFlash weights file '%s': %s", path, strerror(errno));
        return NULL;
    }
    if (fread(len_bytes, 1, sizeof(len_bytes), fp) != sizeof(len_bytes)) {
        dflash_err(err, errlen, "DFlash weights file '%s': missing safetensors header length", path);
        fclose(fp);
        return NULL;
    }
    len = read_le64(len_bytes);
    if (len == 0 || len > 64ull * 1024ull * 1024ull) {
        dflash_err(err, errlen, "DFlash weights file '%s': unsupported safetensors header length", path);
        fclose(fp);
        return NULL;
    }
    if (len > (uint64_t)SIZE_MAX - 1u) {
        dflash_err(err, errlen, "DFlash weights file '%s': safetensors header is too large", path);
        fclose(fp);
        return NULL;
    }
    header = malloc((size_t)len + 1u);
    if (!header) {
        dflash_err(err, errlen, "out of memory reading DFlash safetensors header");
        fclose(fp);
        return NULL;
    }
    if (fread(header, 1, (size_t)len, fp) != (size_t)len) {
        dflash_err(err, errlen, "DFlash weights file '%s': safetensors header read failed", path);
        free(header);
        fclose(fp);
        return NULL;
    }
    header[len] = '\0';
    fclose(fp);
    *header_len = len;
    return header;
}

static uint32_t count_safetensors_tensors(const char *header) {
    uint32_t n = 0;
    const char *p = header;
    while ((p = strstr(p, "\"dtype\"")) != NULL) {
        n++;
        p += 7;
    }
    return n;
}

static int parse_dtype_at(const char *value,
                          dflash_st_dtype *dtype,
                          const char *name,
                          char *err,
                          size_t errlen) {
    const char *p = skip_ws(value);
    const char *q = NULL;
    size_t len = 0;

    if (*p != '"') {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' has non-string dtype",
                          name);
    }
    p++;
    q = strchr(p, '"');
    if (!q) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' has unterminated dtype",
                          name);
    }
    len = (size_t)(q - p);
    if (len == 4 && memcmp(p, "BF16", 4) == 0) {
        *dtype = DFLASH_ST_DTYPE_BF16;
    } else if (len == 4 && memcmp(p, "BOOL", 4) == 0) {
        *dtype = DFLASH_ST_DTYPE_BOOL;
    } else if (len == 3 && memcmp(p, "I64", 3) == 0) {
        *dtype = DFLASH_ST_DTYPE_I64;
    } else {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' has unsupported dtype '%.*s'",
                          name, (int)len, p);
    }
    return 0;
}

static const char *dtype_name(dflash_st_dtype dtype) {
    switch (dtype) {
        case DFLASH_ST_DTYPE_BF16: return "BF16";
        case DFLASH_ST_DTYPE_BOOL: return "BOOL";
        case DFLASH_ST_DTYPE_I64: return "I64";
        default: return "unknown";
    }
}

static int read_tensor_meta(const char *header,
                            const char *name,
                            dflash_tensor_meta *meta,
                            char *err,
                            size_t errlen) {
    const char *value = json_key_value(header, name);
    const char *obj_start = NULL;
    const char *obj_end = NULL;
    const char *dtype_value = NULL;
    const char *shape_value = NULL;

    memset(meta, 0, sizeof(*meta));
    if (!value) {
        return dflash_err(err, errlen,
                          "DFlash safetensors is missing tensor '%s'",
                          name);
    }
    if (json_object_span(value, &obj_start, &obj_end) != 0) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' is not an object",
                          name);
    }
    dtype_value = json_key_value_bounded(obj_start, obj_end, "dtype");
    shape_value = json_key_value_bounded(obj_start, obj_end, "shape");
    if (!dtype_value || !shape_value) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' is missing dtype or shape",
                          name);
    }
    if (parse_dtype_at(dtype_value, &meta->dtype, name, err, errlen) != 0 ||
        parse_u64_array_at(shape_value,
                           meta->shape,
                           (uint32_t)(sizeof(meta->shape) / sizeof(meta->shape[0])),
                           &meta->ndim,
                           name,
                           err,
                           errlen) != 0) {
        return 1;
    }
    return 0;
}

static int expect_tensor(const char *header,
                         const char *name,
                         dflash_st_dtype dtype,
                         const uint64_t *shape,
                         uint32_t ndim,
                         char *err,
                         size_t errlen) {
    dflash_tensor_meta meta;
    if (read_tensor_meta(header, name, &meta, err, errlen) != 0) return 1;
    if (meta.dtype != dtype) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' dtype %s does not match expected %s",
                          name, dtype_name(meta.dtype), dtype_name(dtype));
    }
    if (meta.ndim != ndim) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' rank %u does not match expected %u",
                          name, meta.ndim, ndim);
    }
    for (uint32_t i = 0; i < ndim; i++) {
        if (meta.shape[i] != shape[i]) {
            return dflash_err(err, errlen,
                              "DFlash safetensors tensor '%s' shape[%u]=%llu does not match expected %llu",
                              name,
                              i,
                              (unsigned long long)meta.shape[i],
                              (unsigned long long)shape[i]);
        }
    }
    return 0;
}

static int expect_layer_tensor(const char *header,
                               uint32_t layer,
                               const char *suffix,
                               dflash_st_dtype dtype,
                               const uint64_t *shape,
                               uint32_t ndim,
                               char *err,
                               size_t errlen) {
    char name[160];
    int n = snprintf(name, sizeof(name), "layers.%u.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        return dflash_err(err, errlen, "DFlash safetensors tensor name is too long");
    }
    return expect_tensor(header, name, dtype, shape, ndim, err, errlen);
}

int ds4_dflash_weights_validate(ds4_dflash_weights *w,
                                const char *path,
                                const ds4_dflash_config *cfg,
                                char *err,
                                size_t errlen) {
    char resolved[DS4_DFLASH_MAX_PATH];
    char *header = NULL;
    uint64_t header_len = 0;

    if (!w) return dflash_err(err, errlen, "DFlash weights output is null");
    ds4_dflash_weights_init(w);
    if (!cfg || !cfg->loaded) {
        return dflash_err(err, errlen, "DFlash config must be loaded before weights validation");
    }
    if (cfg->num_hidden_layers == 0 ||
        cfg->intermediate_size == 0 ||
        cfg->num_attention_heads == 0 ||
        cfg->num_key_value_heads == 0 ||
        cfg->head_dim == 0 ||
        cfg->hc_mult == 0) {
        return dflash_err(err, errlen,
                          "DFlash config is missing draft transformer dimensions required for weights validation");
    }
    if (resolve_weights_path(path, resolved, sizeof(resolved), err, errlen) != 0) return 1;
    header = read_safetensors_header(resolved, &header_len, err, errlen);
    if (!header) return 1;

    const uint64_t hidden = cfg->hidden_size;
    const uint64_t target_vocab = cfg->vocab_size;
    const uint64_t draft_vocab = cfg->draft_vocab_size;
    const uint64_t fc_in = (uint64_t)cfg->n_target_layer_ids *
                           (uint64_t)cfg->hc_mult *
                           hidden;
    const uint64_t q_dim = (uint64_t)cfg->num_attention_heads *
                           (uint64_t)cfg->head_dim;
    const uint64_t kv_dim = (uint64_t)cfg->num_key_value_heads *
                            (uint64_t)cfg->head_dim;

    const uint64_t d2t_shape[] = {draft_vocab};
    const uint64_t t2d_shape[] = {target_vocab};
    const uint64_t embed_shape[] = {target_vocab, hidden};
    const uint64_t fc_shape[] = {hidden, fc_in};
    const uint64_t hidden_shape[] = {hidden};
    const uint64_t lm_head_shape[] = {draft_vocab, hidden};
    const uint64_t mlp_in_shape[] = {cfg->intermediate_size, hidden};
    const uint64_t mlp_down_shape[] = {hidden, cfg->intermediate_size};
    const uint64_t q_shape[] = {q_dim, hidden};
    const uint64_t kv_shape[] = {kv_dim, hidden};
    const uint64_t o_shape[] = {hidden, q_dim};
    const uint64_t norm_shape[] = {cfg->head_dim};

    if (expect_tensor(header, "d2t", DFLASH_ST_DTYPE_I64, d2t_shape, 1, err, errlen) != 0 ||
        expect_tensor(header, "t2d", DFLASH_ST_DTYPE_BOOL, t2d_shape, 1, err, errlen) != 0 ||
        expect_tensor(header, "embed_tokens.weight", DFLASH_ST_DTYPE_BF16, embed_shape, 2, err, errlen) != 0 ||
        expect_tensor(header, "fc.weight", DFLASH_ST_DTYPE_BF16, fc_shape, 2, err, errlen) != 0 ||
        expect_tensor(header, "hidden_norm.weight", DFLASH_ST_DTYPE_BF16, hidden_shape, 1, err, errlen) != 0 ||
        expect_tensor(header, "norm.weight", DFLASH_ST_DTYPE_BF16, hidden_shape, 1, err, errlen) != 0 ||
        expect_tensor(header, "lm_head.weight", DFLASH_ST_DTYPE_BF16, lm_head_shape, 2, err, errlen) != 0) {
        free(header);
        return 1;
    }
    for (uint32_t il = 0; il < cfg->num_hidden_layers; il++) {
        if (expect_layer_tensor(header, il, "input_layernorm.weight", DFLASH_ST_DTYPE_BF16, hidden_shape, 1, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "post_attention_layernorm.weight", DFLASH_ST_DTYPE_BF16, hidden_shape, 1, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "mlp.gate_proj.weight", DFLASH_ST_DTYPE_BF16, mlp_in_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "mlp.up_proj.weight", DFLASH_ST_DTYPE_BF16, mlp_in_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "mlp.down_proj.weight", DFLASH_ST_DTYPE_BF16, mlp_down_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "self_attn.q_proj.weight", DFLASH_ST_DTYPE_BF16, q_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "self_attn.k_proj.weight", DFLASH_ST_DTYPE_BF16, kv_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "self_attn.v_proj.weight", DFLASH_ST_DTYPE_BF16, kv_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "self_attn.o_proj.weight", DFLASH_ST_DTYPE_BF16, o_shape, 2, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "self_attn.q_norm.weight", DFLASH_ST_DTYPE_BF16, norm_shape, 1, err, errlen) != 0 ||
            expect_layer_tensor(header, il, "self_attn.k_norm.weight", DFLASH_ST_DTYPE_BF16, norm_shape, 1, err, errlen) != 0) {
            free(header);
            return 1;
        }
    }

    snprintf(w->source_path, sizeof(w->source_path), "%s", resolved);
    w->header_len = header_len;
    w->n_tensors = count_safetensors_tensors(header);
    w->loaded = true;
    free(header);
    return 0;
}
