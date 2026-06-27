#include "ds4_dflash.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS4_DFLASH_RMS_EPS 1.0e-6f
#define DS4_DFLASH_ROPE_THETA 1000000.0f

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
    if (!w) return;
    memset(w, 0, sizeof(*w));
    w->fd = -1;
}

void ds4_dflash_weights_free(ds4_dflash_weights *w) {
    if (!w) return;
    if (w->map && w->file_size) munmap(w->map, (size_t)w->file_size);
    if (w->fd >= 0 && (w->loaded || w->map)) close(w->fd);
    free(w->tensors);
    ds4_dflash_weights_init(w);
}

void ds4_dflash_hidden_history_init(ds4_dflash_hidden_history *h) {
    if (h) memset(h, 0, sizeof(*h));
}

void ds4_dflash_hidden_history_free(ds4_dflash_hidden_history *h) {
    if (!h) return;
    free(h->hidden);
    free(h->positions);
    ds4_dflash_hidden_history_init(h);
}

void ds4_dflash_hidden_history_reset(ds4_dflash_hidden_history *h) {
    if (!h) return;
    h->len = 0;
    h->start = 0;
}

static uint32_t dflash_history_slot(const ds4_dflash_hidden_history *h,
                                    uint32_t logical_row) {
    return (h->start + logical_row) % h->capacity;
}

int ds4_dflash_hidden_history_reserve(ds4_dflash_hidden_history *h,
                                      const ds4_dflash_config *cfg,
                                      uint32_t capacity,
                                      char *err,
                                      size_t errlen) {
    float *hidden = NULL;
    uint32_t *positions = NULL;

    if (!h || !cfg || !cfg->loaded || cfg->hidden_size == 0 || capacity == 0) {
        return dflash_err(err, errlen, "invalid DFlash hidden-history reservation");
    }
    if ((size_t)capacity > SIZE_MAX / sizeof(hidden[0]) / cfg->hidden_size) {
        return dflash_err(err, errlen, "DFlash hidden-history reservation is too large");
    }

    hidden = calloc((size_t)capacity * cfg->hidden_size, sizeof(hidden[0]));
    positions = calloc(capacity, sizeof(positions[0]));
    if (!hidden || !positions) {
        free(hidden);
        free(positions);
        return dflash_err(err, errlen, "out of memory allocating DFlash hidden history");
    }

    ds4_dflash_hidden_history_free(h);
    h->hidden = hidden;
    h->positions = positions;
    h->capacity = capacity;
    h->hidden_size = cfg->hidden_size;
    return 0;
}

int ds4_dflash_hidden_history_append(ds4_dflash_hidden_history *h,
                                     uint32_t position,
                                     const float *hidden,
                                     char *err,
                                     size_t errlen) {
    uint32_t slot = 0;

    if (!h || !h->hidden || !h->positions || h->capacity == 0 ||
        h->hidden_size == 0 || !hidden) {
        return dflash_err(err, errlen, "invalid DFlash hidden-history append");
    }
    if (h->len > 0) {
        const uint32_t last_slot = dflash_history_slot(h, h->len - 1u);
        if (position <= h->positions[last_slot]) {
            return dflash_err(err,
                              errlen,
                              "DFlash hidden-history positions must increase");
        }
    }

    if (h->len < h->capacity) {
        slot = dflash_history_slot(h, h->len);
        h->len++;
    } else {
        slot = h->start;
        h->start = (h->start + 1u) % h->capacity;
    }

    h->positions[slot] = position;
    memcpy(h->hidden + (uint64_t)slot * h->hidden_size,
           hidden,
           (size_t)h->hidden_size * sizeof(h->hidden[0]));
    return 0;
}

uint32_t ds4_dflash_hidden_history_count_visible(const ds4_dflash_hidden_history *h,
                                                 uint32_t anchor_position,
                                                 uint32_t max_rows) {
    uint32_t visible = 0;

    if (!h || !h->positions || h->capacity == 0) return 0;
    for (uint32_t i = 0; i < h->len; i++) {
        const uint32_t slot = dflash_history_slot(h, i);
        if (h->positions[slot] < anchor_position) visible++;
    }
    if (max_rows > 0 && visible > max_rows) visible = max_rows;
    return visible;
}

int ds4_dflash_hidden_history_copy_visible(const ds4_dflash_hidden_history *h,
                                           uint32_t anchor_position,
                                           uint32_t max_rows,
                                           float *target_hidden,
                                           uint32_t *target_positions,
                                           uint32_t *out_rows,
                                           char *err,
                                           size_t errlen) {
    const uint32_t visible = ds4_dflash_hidden_history_count_visible(h,
                                                                     anchor_position,
                                                                     max_rows);
    uint32_t skipped = 0;
    uint32_t copied = 0;
    uint32_t total_visible = 0;

    if (!out_rows) {
        return dflash_err(err, errlen, "DFlash hidden-history output count is missing");
    }
    *out_rows = 0;
    if (!h || !h->hidden || !h->positions || h->capacity == 0 || h->hidden_size == 0) {
        return dflash_err(err, errlen, "invalid DFlash hidden-history copy");
    }
    if (visible > 0 && (!target_hidden || !target_positions)) {
        return dflash_err(err, errlen, "DFlash hidden-history copy outputs are missing");
    }

    for (uint32_t i = 0; i < h->len; i++) {
        const uint32_t slot = dflash_history_slot(h, i);
        if (h->positions[slot] < anchor_position) total_visible++;
    }
    if (total_visible > visible) skipped = total_visible - visible;

    for (uint32_t i = 0; i < h->len && copied < visible; i++) {
        const uint32_t slot = dflash_history_slot(h, i);
        if (h->positions[slot] >= anchor_position) continue;
        if (skipped > 0) {
            skipped--;
            continue;
        }
        target_positions[copied] = h->positions[slot];
        memcpy(target_hidden + (uint64_t)copied * h->hidden_size,
               h->hidden + (uint64_t)slot * h->hidden_size,
               (size_t)h->hidden_size * sizeof(target_hidden[0]));
        copied++;
    }

    *out_rows = copied;
    return 0;
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

typedef struct {
    ds4_dflash_tensor_dtype dtype;
    uint64_t shape[4];
    uint64_t data_offsets[2];
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
                          ds4_dflash_tensor_dtype *dtype,
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
        *dtype = DS4_DFLASH_TENSOR_BF16;
    } else if (len == 4 && memcmp(p, "BOOL", 4) == 0) {
        *dtype = DS4_DFLASH_TENSOR_BOOL;
    } else if (len == 3 && memcmp(p, "I64", 3) == 0) {
        *dtype = DS4_DFLASH_TENSOR_I64;
    } else {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' has unsupported dtype '%.*s'",
                          name, (int)len, p);
    }
    return 0;
}

static const char *dtype_name(ds4_dflash_tensor_dtype dtype) {
    switch (dtype) {
        case DS4_DFLASH_TENSOR_BF16: return "BF16";
        case DS4_DFLASH_TENSOR_BOOL: return "BOOL";
        case DS4_DFLASH_TENSOR_I64: return "I64";
        default: return "unknown";
    }
}

static uint64_t dtype_size(ds4_dflash_tensor_dtype dtype) {
    switch (dtype) {
        case DS4_DFLASH_TENSOR_BF16: return 2;
        case DS4_DFLASH_TENSOR_BOOL: return 1;
        case DS4_DFLASH_TENSOR_I64: return 8;
        default: return 0;
    }
}

static int tensor_expected_bytes(ds4_dflash_tensor_dtype dtype,
                                 const uint64_t *shape,
                                 uint32_t ndim,
                                 uint64_t *out,
                                 char *err,
                                 size_t errlen) {
    uint64_t n = 1;
    const uint64_t elem = dtype_size(dtype);
    if (elem == 0 || ndim == 0) {
        return dflash_err(err, errlen, "DFlash safetensors tensor has invalid dtype or rank");
    }
    for (uint32_t i = 0; i < ndim; i++) {
        if (shape[i] != 0 && n > UINT64_MAX / shape[i]) {
            return dflash_err(err, errlen, "DFlash safetensors tensor shape overflows");
        }
        n *= shape[i];
    }
    if (n > UINT64_MAX / elem) {
        return dflash_err(err, errlen, "DFlash safetensors tensor byte size overflows");
    }
    *out = n * elem;
    return 0;
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
    const char *offsets_value = NULL;
    uint32_t n_offsets = 0;

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
    offsets_value = json_key_value_bounded(obj_start, obj_end, "data_offsets");
    if (!dtype_value || !shape_value || !offsets_value) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' is missing dtype, shape, or data_offsets",
                          name);
    }
    if (parse_dtype_at(dtype_value, &meta->dtype, name, err, errlen) != 0 ||
        parse_u64_array_at(shape_value,
                           meta->shape,
                           (uint32_t)(sizeof(meta->shape) / sizeof(meta->shape[0])),
                           &meta->ndim,
                           name,
                           err,
                           errlen) != 0 ||
        parse_u64_array_at(offsets_value,
                           meta->data_offsets,
                           2,
                           &n_offsets,
                           name,
                           err,
                           errlen) != 0) {
        return 1;
    }
    if (n_offsets != 2 || meta->data_offsets[1] < meta->data_offsets[0]) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' has invalid data_offsets",
                          name);
    }
    return 0;
}

static int expect_tensor(const char *header,
                         const char *name,
                         ds4_dflash_tensor_dtype dtype,
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
                               ds4_dflash_tensor_dtype dtype,
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

static int bind_tensor(ds4_dflash_weights *w,
                       const char *header,
                       const char *name,
                       ds4_dflash_tensor_dtype dtype,
                       const uint64_t *shape,
                       uint32_t ndim,
                       char *err,
                       size_t errlen) {
    dflash_tensor_meta meta;
    uint64_t expected_bytes = 0;
    ds4_dflash_tensor *t = NULL;
    int n = 0;

    if (!w || !w->tensors || w->n_bound_tensors >= w->n_tensors) {
        return dflash_err(err, errlen, "DFlash safetensors tensor table is full");
    }
    if (expect_tensor(header, name, dtype, shape, ndim, err, errlen) != 0 ||
        read_tensor_meta(header, name, &meta, err, errlen) != 0 ||
        tensor_expected_bytes(dtype, shape, ndim, &expected_bytes, err, errlen) != 0) {
        return 1;
    }
    if (meta.data_offsets[1] - meta.data_offsets[0] != expected_bytes) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' has %llu bytes, expected %llu",
                          name,
                          (unsigned long long)(meta.data_offsets[1] - meta.data_offsets[0]),
                          (unsigned long long)expected_bytes);
    }
    if (w->data_start > UINT64_MAX - meta.data_offsets[0] ||
        w->data_start + meta.data_offsets[1] > w->file_size) {
        return dflash_err(err, errlen,
                          "DFlash safetensors tensor '%s' extends past mapped file",
                          name);
    }

    t = &w->tensors[w->n_bound_tensors++];
    memset(t, 0, sizeof(*t));
    n = snprintf(t->name, sizeof(t->name), "%s", name);
    if (n < 0 || (size_t)n >= sizeof(t->name)) {
        return dflash_err(err, errlen, "DFlash safetensors tensor name is too long");
    }
    t->dtype = dtype;
    t->ndim = ndim;
    memcpy(t->shape, shape, (size_t)ndim * sizeof(t->shape[0]));
    t->data_offsets[0] = meta.data_offsets[0];
    t->data_offsets[1] = meta.data_offsets[1];
    t->abs_offset = w->data_start + meta.data_offsets[0];
    t->nbytes = expected_bytes;
    return 0;
}

static int bind_layer_tensor(ds4_dflash_weights *w,
                             const char *header,
                             uint32_t layer,
                             const char *suffix,
                             ds4_dflash_tensor_dtype dtype,
                             const uint64_t *shape,
                             uint32_t ndim,
                             char *err,
                             size_t errlen) {
    char name[160];
    int n = snprintf(name, sizeof(name), "layers.%u.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        return dflash_err(err, errlen, "DFlash safetensors tensor name is too long");
    }
    return bind_tensor(w, header, name, dtype, shape, ndim, err, errlen);
}

static int visit_tensor(ds4_dflash_weights *w,
                        const char *header,
                        const char *name,
                        ds4_dflash_tensor_dtype dtype,
                        const uint64_t *shape,
                        uint32_t ndim,
                        char *err,
                        size_t errlen) {
    if (w) return bind_tensor(w, header, name, dtype, shape, ndim, err, errlen);
    return expect_tensor(header, name, dtype, shape, ndim, err, errlen);
}

static int visit_layer_tensor(ds4_dflash_weights *w,
                              const char *header,
                              uint32_t layer,
                              const char *suffix,
                              ds4_dflash_tensor_dtype dtype,
                              const uint64_t *shape,
                              uint32_t ndim,
                              char *err,
                              size_t errlen) {
    if (w) return bind_layer_tensor(w, header, layer, suffix, dtype, shape, ndim, err, errlen);
    return expect_layer_tensor(header, layer, suffix, dtype, shape, ndim, err, errlen);
}

static int visit_required_tensors(ds4_dflash_weights *w,
                                  const char *header,
                                  const ds4_dflash_config *cfg,
                                  char *err,
                                  size_t errlen) {
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

    if (visit_tensor(w, header, "d2t", DS4_DFLASH_TENSOR_I64, d2t_shape, 1, err, errlen) != 0 ||
        visit_tensor(w, header, "t2d", DS4_DFLASH_TENSOR_BOOL, t2d_shape, 1, err, errlen) != 0 ||
        visit_tensor(w, header, "embed_tokens.weight", DS4_DFLASH_TENSOR_BF16, embed_shape, 2, err, errlen) != 0 ||
        visit_tensor(w, header, "fc.weight", DS4_DFLASH_TENSOR_BF16, fc_shape, 2, err, errlen) != 0 ||
        visit_tensor(w, header, "hidden_norm.weight", DS4_DFLASH_TENSOR_BF16, hidden_shape, 1, err, errlen) != 0 ||
        visit_tensor(w, header, "norm.weight", DS4_DFLASH_TENSOR_BF16, hidden_shape, 1, err, errlen) != 0 ||
        visit_tensor(w, header, "lm_head.weight", DS4_DFLASH_TENSOR_BF16, lm_head_shape, 2, err, errlen) != 0) {
        return 1;
    }
    for (uint32_t il = 0; il < cfg->num_hidden_layers; il++) {
        if (visit_layer_tensor(w, header, il, "input_layernorm.weight", DS4_DFLASH_TENSOR_BF16, hidden_shape, 1, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "post_attention_layernorm.weight", DS4_DFLASH_TENSOR_BF16, hidden_shape, 1, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "mlp.gate_proj.weight", DS4_DFLASH_TENSOR_BF16, mlp_in_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "mlp.up_proj.weight", DS4_DFLASH_TENSOR_BF16, mlp_in_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "mlp.down_proj.weight", DS4_DFLASH_TENSOR_BF16, mlp_down_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "self_attn.q_proj.weight", DS4_DFLASH_TENSOR_BF16, q_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "self_attn.k_proj.weight", DS4_DFLASH_TENSOR_BF16, kv_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "self_attn.v_proj.weight", DS4_DFLASH_TENSOR_BF16, kv_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "self_attn.o_proj.weight", DS4_DFLASH_TENSOR_BF16, o_shape, 2, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "self_attn.q_norm.weight", DS4_DFLASH_TENSOR_BF16, norm_shape, 1, err, errlen) != 0 ||
            visit_layer_tensor(w, header, il, "self_attn.k_norm.weight", DS4_DFLASH_TENSOR_BF16, norm_shape, 1, err, errlen) != 0) {
            return 1;
        }
    }
    return 0;
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

    if (visit_required_tensors(NULL, header, cfg, err, errlen) != 0) {
        free(header);
        return 1;
    }

    snprintf(w->source_path, sizeof(w->source_path), "%s", resolved);
    w->header_len = header_len;
    w->n_tensors = count_safetensors_tensors(header);
    w->loaded = true;
    free(header);
    return 0;
}

int ds4_dflash_weights_open(ds4_dflash_weights *w,
                            const char *path,
                            const ds4_dflash_config *cfg,
                            char *err,
                            size_t errlen) {
    char resolved[DS4_DFLASH_MAX_PATH];
    char *header = NULL;
    uint64_t header_len = 0;
    struct stat st;
    int fd = -1;
    void *map = NULL;
    uint32_t n_tensors = 0;

    if (!w) return dflash_err(err, errlen, "DFlash weights output is null");
    ds4_dflash_weights_init(w);
    if (!cfg || !cfg->loaded) {
        return dflash_err(err, errlen, "DFlash config must be loaded before weights open");
    }
    if (cfg->num_hidden_layers == 0 ||
        cfg->intermediate_size == 0 ||
        cfg->num_attention_heads == 0 ||
        cfg->num_key_value_heads == 0 ||
        cfg->head_dim == 0 ||
        cfg->hc_mult == 0) {
        return dflash_err(err, errlen,
                          "DFlash config is missing draft transformer dimensions required for weights open");
    }
    if (resolve_weights_path(path, resolved, sizeof(resolved), err, errlen) != 0) return 1;
    header = read_safetensors_header(resolved, &header_len, err, errlen);
    if (!header) return 1;
    n_tensors = count_safetensors_tensors(header);
    fd = open(resolved, O_RDONLY);
    if (fd < 0) {
        free(header);
        return dflash_err(err, errlen, "DFlash weights file '%s': %s", resolved, strerror(errno));
    }
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DFlash weights file '%s': stat failed", resolved);
    }
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DFlash weights file '%s' is too large to map", resolved);
    }
    if ((uint64_t)st.st_size < 8u + header_len) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DFlash weights file '%s' is shorter than its header", resolved);
    }
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DFlash weights file '%s': mmap failed: %s", resolved, strerror(errno));
    }

    snprintf(w->source_path, sizeof(w->source_path), "%s", resolved);
    w->header_len = header_len;
    w->file_size = (uint64_t)st.st_size;
    w->data_start = 8u + header_len;
    w->map = map;
    w->fd = fd;
    w->n_tensors = n_tensors;
    w->tensors = calloc(n_tensors ? n_tensors : 1u, sizeof(w->tensors[0]));
    if (!w->tensors) {
        ds4_dflash_weights_free(w);
        free(header);
        return dflash_err(err, errlen, "out of memory binding DFlash safetensors");
    }
    if (visit_required_tensors(w, header, cfg, err, errlen) != 0) {
        ds4_dflash_weights_free(w);
        free(header);
        return 1;
    }
    w->loaded = true;
    free(header);
    return 0;
}

const ds4_dflash_tensor *ds4_dflash_weights_find_tensor(const ds4_dflash_weights *w,
                                                        const char *name) {
    if (!w || !name) return NULL;
    for (uint32_t i = 0; i < w->n_bound_tensors; i++) {
        if (strcmp(w->tensors[i].name, name) == 0) return &w->tensors[i];
    }
    return NULL;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float bf16_data_at(const ds4_dflash_weights *w,
                          const ds4_dflash_tensor *tensor,
                          uint64_t elem) {
    const unsigned char *src = (const unsigned char *)w->map +
                               tensor->abs_offset +
                               elem * 2u;
    const uint16_t raw = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
    return bf16_to_f32(raw);
}

static int64_t i64_data_at(const ds4_dflash_weights *w,
                           const ds4_dflash_tensor *tensor,
                           uint64_t elem) {
    const unsigned char *src = (const unsigned char *)w->map +
                               tensor->abs_offset +
                               elem * 8u;
    uint64_t raw = 0;
    for (uint32_t i = 0; i < 8; i++) {
        raw |= (uint64_t)src[i] << (8u * i);
    }
    return (int64_t)raw;
}

static bool bool_data_at(const ds4_dflash_weights *w,
                         const ds4_dflash_tensor *tensor,
                         uint64_t elem) {
    const unsigned char *src = (const unsigned char *)w->map +
                               tensor->abs_offset +
                               elem;
    return src[0] != 0;
}

int ds4_dflash_tensor_read_bf16_f32(const ds4_dflash_weights *w,
                                    const ds4_dflash_tensor *tensor,
                                    uint64_t elem_offset,
                                    float *out,
                                    uint64_t n,
                                    char *err,
                                    size_t errlen) {
    if (!w || !w->loaded || !w->map || !tensor || !out) {
        return dflash_err(err, errlen, "DFlash BF16 read has invalid input");
    }
    if (tensor->dtype != DS4_DFLASH_TENSOR_BF16) {
        return dflash_err(err, errlen,
                          "DFlash tensor '%s' is %s, not BF16",
                          tensor->name, dtype_name(tensor->dtype));
    }
    if (elem_offset > UINT64_MAX / 2u || n > UINT64_MAX / 2u ||
        elem_offset * 2u > tensor->nbytes ||
        n * 2u > tensor->nbytes - elem_offset * 2u) {
        return dflash_err(err, errlen,
                          "DFlash BF16 read for tensor '%s' is out of bounds",
                          tensor->name);
    }
    const unsigned char *src = (const unsigned char *)w->map +
                               tensor->abs_offset +
                               elem_offset * 2u;
    for (uint64_t i = 0; i < n; i++) {
        const uint16_t raw = (uint16_t)src[i * 2u] |
                             ((uint16_t)src[i * 2u + 1u] << 8);
        out[i] = bf16_to_f32(raw);
    }
    return 0;
}

int ds4_dflash_weights_read_bf16_f32(const ds4_dflash_weights *w,
                                     const char *name,
                                     uint64_t elem_offset,
                                     float *out,
                                     uint64_t n,
                                     char *err,
                                     size_t errlen) {
    const ds4_dflash_tensor *tensor = ds4_dflash_weights_find_tensor(w, name);
    if (!tensor) {
        return dflash_err(err, errlen,
                          "DFlash safetensors is missing bound tensor '%s'",
                          name ? name : "(null)");
    }
    return ds4_dflash_tensor_read_bf16_f32(w, tensor, elem_offset, out, n, err, errlen);
}

static int layer_tensor_name(char *out,
                             size_t outlen,
                             uint32_t layer,
                             const char *suffix,
                             char *err,
                             size_t errlen) {
    int n = snprintf(out, outlen, "layers.%u.%s", layer, suffix);
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "DFlash layer tensor name is too long");
    }
    return 0;
}

static int require_bf16_tensor_1d(const ds4_dflash_weights *w,
                                  const char *name,
                                  uint64_t d0,
                                  const ds4_dflash_tensor **out,
                                  char *err,
                                  size_t errlen) {
    const ds4_dflash_tensor *tensor = ds4_dflash_weights_find_tensor(w, name);
    if (!tensor) {
        return dflash_err(err, errlen,
                          "DFlash safetensors is missing bound tensor '%s'",
                          name);
    }
    if (tensor->dtype != DS4_DFLASH_TENSOR_BF16 ||
        tensor->ndim != 1 ||
        tensor->shape[0] != d0) {
        return dflash_err(err, errlen,
                          "DFlash tensor '%s' layout does not match CPU draft graph",
                          name);
    }
    *out = tensor;
    return 0;
}

static int require_bf16_tensor_2d(const ds4_dflash_weights *w,
                                  const char *name,
                                  uint64_t d0,
                                  uint64_t d1,
                                  const ds4_dflash_tensor **out,
                                  char *err,
                                  size_t errlen) {
    const ds4_dflash_tensor *tensor = ds4_dflash_weights_find_tensor(w, name);
    if (!tensor) {
        return dflash_err(err, errlen,
                          "DFlash safetensors is missing bound tensor '%s'",
                          name);
    }
    if (tensor->dtype != DS4_DFLASH_TENSOR_BF16 ||
        tensor->ndim != 2 ||
        tensor->shape[0] != d0 ||
        tensor->shape[1] != d1) {
        return dflash_err(err, errlen,
                          "DFlash tensor '%s' layout does not match CPU draft graph",
                          name);
    }
    *out = tensor;
    return 0;
}

static int require_tensor_1d(const ds4_dflash_weights *w,
                             const char *name,
                             ds4_dflash_tensor_dtype dtype,
                             uint64_t d0,
                             const ds4_dflash_tensor **out,
                             char *err,
                             size_t errlen) {
    const ds4_dflash_tensor *tensor = ds4_dflash_weights_find_tensor(w, name);
    if (!tensor) {
        return dflash_err(err, errlen,
                          "DFlash safetensors is missing bound tensor '%s'",
                          name);
    }
    if (tensor->dtype != dtype ||
        tensor->ndim != 1 ||
        tensor->shape[0] != d0) {
        return dflash_err(err, errlen,
                          "DFlash tensor '%s' layout does not match CPU draft graph",
                          name);
    }
    *out = tensor;
    return 0;
}

static void rms_norm_bf16_weight(const ds4_dflash_weights *w,
                                 const ds4_dflash_tensor *weight,
                                 const float *in,
                                 uint64_t n,
                                 float *out) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        ss += (double)in[i] * (double)in[i];
    }
    const float inv_rms = 1.0f / sqrtf((float)(ss / (double)n) + DS4_DFLASH_RMS_EPS);
    for (uint64_t i = 0; i < n; i++) {
        out[i] = in[i] * inv_rms * bf16_data_at(w, weight, i);
    }
}

static void linear_bf16(const ds4_dflash_weights *w,
                        const ds4_dflash_tensor *weight,
                        const float *in,
                        float *out) {
    const uint64_t out_dim = weight->shape[0];
    const uint64_t in_dim = weight->shape[1];
    for (uint64_t o = 0; o < out_dim; o++) {
        double acc = 0.0;
        const uint64_t row = o * in_dim;
        for (uint64_t i = 0; i < in_dim; i++) {
            acc += (double)in[i] * (double)bf16_data_at(w, weight, row + i);
        }
        out[o] = (float)acc;
    }
}

static float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static void rope_head_qwen3_inplace(float *x, uint32_t head_dim, uint32_t pos) {
    const uint32_t half = head_dim / 2u;
    for (uint32_t i = 0; i < half; i++) {
        const float theta =
            powf(DS4_DFLASH_ROPE_THETA, -((float)(2u * i) / (float)head_dim));
        const float angle = (float)pos * theta;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float a = x[i];
        const float b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

static float dot_f32_local(const float *a, const float *b, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

static int checked_mul_u64(uint64_t a,
                           uint64_t b,
                           uint64_t *out,
                           const char *what,
                           char *err,
                           size_t errlen) {
    if (a != 0 && b > UINT64_MAX / a) {
        return dflash_err(err, errlen, "DFlash CPU %s is too large", what);
    }
    *out = a * b;
    return 0;
}

static int alloc_f32(float **out,
                     uint64_t n,
                     const char *what,
                     char *err,
                     size_t errlen) {
    if (n > SIZE_MAX / sizeof(float)) {
        return dflash_err(err, errlen, "DFlash CPU %s is too large", what);
    }
    *out = calloc((size_t)n, sizeof(float));
    if (!*out) {
        return dflash_err(err, errlen, "out of memory allocating DFlash CPU %s", what);
    }
    return 0;
}

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
                                  size_t errlen) {
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t q_dim = cfg ? (uint64_t)cfg->num_attention_heads * cfg->head_dim : 0;
    const uint64_t kv_dim = cfg ? (uint64_t)cfg->num_key_value_heads * cfg->head_dim : 0;
    const uint64_t total_kv_rows = (uint64_t)n_target_rows + (uint64_t)n_noise_rows;
    const ds4_dflash_tensor *input_norm = NULL;
    const ds4_dflash_tensor *q_proj = NULL;
    const ds4_dflash_tensor *k_proj = NULL;
    const ds4_dflash_tensor *v_proj = NULL;
    const ds4_dflash_tensor *o_proj = NULL;
    const ds4_dflash_tensor *q_norm = NULL;
    const ds4_dflash_tensor *k_norm = NULL;
    char name[DS4_DFLASH_MAX_TENSOR_NAME];
    float *norm_noise = NULL;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *heads = NULL;
    float *scores = NULL;
    uint64_t norm_noise_n = 0;
    uint64_t q_n = 0;
    uint64_t kv_n = 0;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !noise_hidden ||
        !noise_positions || !out || n_noise_rows == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU attention request");
    }
    if (n_target_rows > 0 && (!target_hidden || !target_positions)) {
        return dflash_err(err, errlen, "DFlash CPU attention target rows are missing");
    }
    if (layer >= cfg->num_hidden_layers) {
        return dflash_err(err, errlen,
                          "DFlash CPU attention layer %u is outside configured draft depth %u",
                          layer,
                          cfg->num_hidden_layers);
    }
    if (hidden == 0 || q_dim == 0 || kv_dim == 0 ||
        cfg->head_dim == 0 ||
        cfg->head_dim % 2u != 0 ||
        cfg->num_attention_heads == 0 ||
        cfg->num_key_value_heads == 0 ||
        cfg->num_attention_heads % cfg->num_key_value_heads != 0) {
        return dflash_err(err, errlen, "DFlash CPU attention config is missing dimensions");
    }
    if (total_kv_rows < n_noise_rows) {
        return dflash_err(err, errlen, "DFlash CPU attention row count overflow");
    }

    if (layer_tensor_name(name, sizeof(name), layer, "input_layernorm.weight", err, errlen) != 0 ||
        require_bf16_tensor_1d(w, name, hidden, &input_norm, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "self_attn.q_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, q_dim, hidden, &q_proj, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "self_attn.k_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, kv_dim, hidden, &k_proj, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "self_attn.v_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, kv_dim, hidden, &v_proj, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "self_attn.o_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, hidden, q_dim, &o_proj, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "self_attn.q_norm.weight", err, errlen) != 0 ||
        require_bf16_tensor_1d(w, name, cfg->head_dim, &q_norm, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "self_attn.k_norm.weight", err, errlen) != 0 ||
        require_bf16_tensor_1d(w, name, cfg->head_dim, &k_norm, err, errlen) != 0) {
        return 1;
    }

    if (checked_mul_u64(n_noise_rows, hidden, &norm_noise_n, "noise span", err, errlen) != 0 ||
        checked_mul_u64(n_noise_rows, q_dim, &q_n, "query span", err, errlen) != 0 ||
        checked_mul_u64(total_kv_rows, kv_dim, &kv_n, "kv span", err, errlen) != 0 ||
        alloc_f32(&norm_noise, norm_noise_n, "noise norm", err, errlen) != 0 ||
        alloc_f32(&q, q_n, "queries", err, errlen) != 0 ||
        alloc_f32(&k, kv_n, "keys", err, errlen) != 0 ||
        alloc_f32(&v, kv_n, "values", err, errlen) != 0 ||
        alloc_f32(&heads, q_n, "heads", err, errlen) != 0 ||
        alloc_f32(&scores, total_kv_rows, "scores", err, errlen) != 0) {
        goto done;
    }

    for (uint32_t row = 0; row < n_noise_rows; row++) {
        const float *x = noise_hidden + (uint64_t)row * hidden;
        float *normed = norm_noise + (uint64_t)row * hidden;
        float *qr = q + (uint64_t)row * q_dim;
        rms_norm_bf16_weight(w, input_norm, x, hidden, normed);
        linear_bf16(w, q_proj, normed, qr);
        for (uint32_t h = 0; h < cfg->num_attention_heads; h++) {
            float *head = qr + (uint64_t)h * cfg->head_dim;
            rms_norm_bf16_weight(w, q_norm, head, cfg->head_dim, head);
            rope_head_qwen3_inplace(head, cfg->head_dim, noise_positions[row]);
        }
    }

    for (uint32_t row = 0; row < n_target_rows; row++) {
        const float *x = target_hidden + (uint64_t)row * hidden;
        float *kr = k + (uint64_t)row * kv_dim;
        float *vr = v + (uint64_t)row * kv_dim;
        linear_bf16(w, k_proj, x, kr);
        linear_bf16(w, v_proj, x, vr);
        for (uint32_t h = 0; h < cfg->num_key_value_heads; h++) {
            float *head = kr + (uint64_t)h * cfg->head_dim;
            rms_norm_bf16_weight(w, k_norm, head, cfg->head_dim, head);
            rope_head_qwen3_inplace(head, cfg->head_dim, target_positions[row]);
        }
    }
    for (uint32_t row = 0; row < n_noise_rows; row++) {
        const uint64_t kv_row = (uint64_t)n_target_rows + row;
        const float *x = norm_noise + (uint64_t)row * hidden;
        float *kr = k + kv_row * kv_dim;
        float *vr = v + kv_row * kv_dim;
        linear_bf16(w, k_proj, x, kr);
        linear_bf16(w, v_proj, x, vr);
        for (uint32_t h = 0; h < cfg->num_key_value_heads; h++) {
            float *head = kr + (uint64_t)h * cfg->head_dim;
            rms_norm_bf16_weight(w, k_norm, head, cfg->head_dim, head);
            rope_head_qwen3_inplace(head, cfg->head_dim, noise_positions[row]);
        }
    }

    const float scale = 1.0f / sqrtf((float)cfg->head_dim);
    const uint32_t kv_groups = cfg->num_attention_heads / cfg->num_key_value_heads;
    for (uint32_t row = 0; row < n_noise_rows; row++) {
        for (uint32_t h = 0; h < cfg->num_attention_heads; h++) {
            const uint32_t kv_head = h / kv_groups;
            const float *qh = q + (uint64_t)row * q_dim + (uint64_t)h * cfg->head_dim;
            float *oh = heads + (uint64_t)row * q_dim + (uint64_t)h * cfg->head_dim;
            float max_score = -FLT_MAX;
            for (uint64_t kr = 0; kr < total_kv_rows; kr++) {
                const float *kh = k + kr * kv_dim + (uint64_t)kv_head * cfg->head_dim;
                scores[kr] = dot_f32_local(qh, kh, cfg->head_dim) * scale;
                if (scores[kr] > max_score) max_score = scores[kr];
            }
            memset(oh, 0, (size_t)cfg->head_dim * sizeof(oh[0]));
            float denom = 0.0f;
            for (uint64_t kr = 0; kr < total_kv_rows; kr++) {
                const float weight = expf(scores[kr] - max_score);
                const float *vh = v + kr * kv_dim + (uint64_t)kv_head * cfg->head_dim;
                denom += weight;
                for (uint32_t i = 0; i < cfg->head_dim; i++) {
                    oh[i] += weight * vh[i];
                }
            }
            const float inv = 1.0f / denom;
            for (uint32_t i = 0; i < cfg->head_dim; i++) oh[i] *= inv;
        }
        linear_bf16(w, o_proj, heads + (uint64_t)row * q_dim, out + (uint64_t)row * hidden);
        for (uint64_t i = 0; i < hidden; i++) {
            out[(uint64_t)row * hidden + i] += noise_hidden[(uint64_t)row * hidden + i];
        }
    }

    rc = 0;

done:
    free(norm_noise);
    free(q);
    free(k);
    free(v);
    free(heads);
    free(scores);
    return rc;
}

int ds4_dflash_cpu_eval_mlp(const ds4_dflash_weights *w,
                            const ds4_dflash_config *cfg,
                            uint32_t layer,
                            const float *hidden_states,
                            uint32_t n_rows,
                            float *out,
                            char *err,
                            size_t errlen) {
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t intermediate = cfg ? cfg->intermediate_size : 0;
    const ds4_dflash_tensor *post_norm = NULL;
    const ds4_dflash_tensor *gate_proj = NULL;
    const ds4_dflash_tensor *up_proj = NULL;
    const ds4_dflash_tensor *down_proj = NULL;
    char name[DS4_DFLASH_MAX_TENSOR_NAME];
    float *normed = NULL;
    float *gate = NULL;
    float *up = NULL;
    float *mid = NULL;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !hidden_states || !out || n_rows == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU MLP request");
    }
    if (layer >= cfg->num_hidden_layers) {
        return dflash_err(err, errlen,
                          "DFlash CPU MLP layer %u is outside configured draft depth %u",
                          layer,
                          cfg->num_hidden_layers);
    }
    if (hidden == 0 || intermediate == 0) {
        return dflash_err(err, errlen, "DFlash CPU MLP config is missing dimensions");
    }
    if ((uint64_t)n_rows > UINT64_MAX / hidden) {
        return dflash_err(err, errlen, "DFlash CPU MLP row span is too large");
    }

    if (layer_tensor_name(name, sizeof(name), layer, "post_attention_layernorm.weight", err, errlen) != 0 ||
        require_bf16_tensor_1d(w, name, hidden, &post_norm, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "mlp.gate_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, intermediate, hidden, &gate_proj, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "mlp.up_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, intermediate, hidden, &up_proj, err, errlen) != 0 ||
        layer_tensor_name(name, sizeof(name), layer, "mlp.down_proj.weight", err, errlen) != 0 ||
        require_bf16_tensor_2d(w, name, hidden, intermediate, &down_proj, err, errlen) != 0) {
        return 1;
    }

    if (hidden > SIZE_MAX / sizeof(float) ||
        intermediate > SIZE_MAX / sizeof(float)) {
        return dflash_err(err, errlen, "DFlash CPU MLP dimensions are too large");
    }
    normed = malloc((size_t)hidden * sizeof(float));
    gate = malloc((size_t)intermediate * sizeof(float));
    up = malloc((size_t)intermediate * sizeof(float));
    mid = malloc((size_t)intermediate * sizeof(float));
    if (!normed || !gate || !up || !mid) {
        free(normed);
        free(gate);
        free(up);
        free(mid);
        return dflash_err(err, errlen, "out of memory evaluating DFlash CPU MLP");
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        const float *x = hidden_states + (uint64_t)row * hidden;
        float *y = out + (uint64_t)row * hidden;
        rms_norm_bf16_weight(w, post_norm, x, hidden, normed);
        linear_bf16(w, gate_proj, normed, gate);
        linear_bf16(w, up_proj, normed, up);
        for (uint64_t i = 0; i < intermediate; i++) {
            mid[i] = silu_f32(gate[i]) * up[i];
        }
        linear_bf16(w, down_proj, mid, y);
        for (uint64_t i = 0; i < hidden; i++) {
            y[i] += x[i];
        }
    }

    free(normed);
    free(gate);
    free(up);
    free(mid);
    return 0;
}

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
                              size_t errlen) {
    uint64_t span = 0;
    float *attn = NULL;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !noise_hidden ||
        !noise_positions || !out || n_noise_rows == 0 || cfg->hidden_size == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU layer request");
    }
    if (checked_mul_u64(n_noise_rows, cfg->hidden_size, &span, "layer span", err, errlen) != 0 ||
        alloc_f32(&attn, span, "layer attention", err, errlen) != 0) {
        return 1;
    }

    if (ds4_dflash_cpu_eval_attention(w,
                                      cfg,
                                      layer,
                                      target_hidden,
                                      target_positions,
                                      n_target_rows,
                                      noise_hidden,
                                      noise_positions,
                                      n_noise_rows,
                                      attn,
                                      err,
                                      errlen) != 0) {
        goto done;
    }
    if (ds4_dflash_cpu_eval_mlp(w,
                                cfg,
                                layer,
                                attn,
                                n_noise_rows,
                                out,
                                err,
                                errlen) != 0) {
        goto done;
    }
    rc = 0;

done:
    free(attn);
    return rc;
}

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
                              size_t errlen) {
    uint64_t span = 0;
    float *cur = NULL;
    float *next = NULL;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !noise_hidden ||
        !noise_positions || !out || n_noise_rows == 0 ||
        cfg->hidden_size == 0 || cfg->num_hidden_layers == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU block request");
    }
    if (checked_mul_u64(n_noise_rows, cfg->hidden_size, &span, "block span", err, errlen) != 0 ||
        alloc_f32(&cur, span, "block current", err, errlen) != 0 ||
        alloc_f32(&next, span, "block next", err, errlen) != 0) {
        goto done;
    }
    memcpy(cur, noise_hidden, (size_t)span * sizeof(cur[0]));

    for (uint32_t layer = 0; layer < cfg->num_hidden_layers; layer++) {
        float *tmp = NULL;
        if (ds4_dflash_cpu_eval_layer(w,
                                      cfg,
                                      layer,
                                      target_hidden,
                                      target_positions,
                                      n_target_rows,
                                      cur,
                                      noise_positions,
                                      n_noise_rows,
                                      next,
                                      err,
                                      errlen) != 0) {
            goto done;
        }
        tmp = cur;
        cur = next;
        next = tmp;
    }

    memcpy(out, cur, (size_t)span * sizeof(out[0]));
    rc = 0;

done:
    free(cur);
    free(next);
    return rc;
}

int ds4_dflash_cpu_eval_logits(const ds4_dflash_weights *w,
                               const ds4_dflash_config *cfg,
                               const float *hidden_states,
                               uint32_t n_rows,
                               float *logits,
                               char *err,
                               size_t errlen) {
    const ds4_dflash_tensor *norm = NULL;
    const ds4_dflash_tensor *lm_head = NULL;
    float *normed = NULL;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !hidden_states ||
        !logits || n_rows == 0 || cfg->hidden_size == 0 ||
        cfg->draft_vocab_size == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU logits request");
    }
    if (require_bf16_tensor_1d(w, "norm.weight", cfg->hidden_size, &norm, err, errlen) != 0 ||
        require_bf16_tensor_2d(w,
                               "lm_head.weight",
                               cfg->draft_vocab_size,
                               cfg->hidden_size,
                               &lm_head,
                               err,
                               errlen) != 0 ||
        alloc_f32(&normed, cfg->hidden_size, "logits norm", err, errlen) != 0) {
        return 1;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        rms_norm_bf16_weight(w,
                             norm,
                             hidden_states + (uint64_t)row * cfg->hidden_size,
                             cfg->hidden_size,
                             normed);
        linear_bf16(w,
                    lm_head,
                    normed,
                    logits + (uint64_t)row * cfg->draft_vocab_size);
    }
    rc = 0;

    free(normed);
    return rc;
}

int ds4_dflash_cpu_select_tokens(const ds4_dflash_weights *w,
                                 const ds4_dflash_config *cfg,
                                 const float *logits,
                                 uint32_t n_rows,
                                 uint32_t *draft_tokens,
                                 uint32_t *target_tokens,
                                 char *err,
                                 size_t errlen) {
    const ds4_dflash_tensor *d2t = NULL;
    const ds4_dflash_tensor *t2d = NULL;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !logits ||
        !draft_tokens || !target_tokens || n_rows == 0 ||
        cfg->draft_vocab_size == 0 || cfg->vocab_size == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU token selection request");
    }
    if (require_tensor_1d(w, "d2t", DS4_DFLASH_TENSOR_I64, cfg->draft_vocab_size, &d2t, err, errlen) != 0 ||
        require_tensor_1d(w, "t2d", DS4_DFLASH_TENSOR_BOOL, cfg->vocab_size, &t2d, err, errlen) != 0) {
        return 1;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        const float *row_logits = logits + (uint64_t)row * cfg->draft_vocab_size;
        uint32_t best = 0;
        float best_score = row_logits[0];
        for (uint32_t tok = 1; tok < cfg->draft_vocab_size; tok++) {
            if (row_logits[tok] > best_score) {
                best_score = row_logits[tok];
                best = tok;
            }
        }
        const int64_t mapped = i64_data_at(w, d2t, best);
        if (mapped < 0 || (uint64_t)mapped >= cfg->vocab_size) {
            return dflash_err(err, errlen,
                              "DFlash draft token %u maps outside target vocab",
                              best);
        }
        if (!bool_data_at(w, t2d, (uint64_t)mapped)) {
            return dflash_err(err, errlen,
                              "DFlash draft token %u maps to inadmissible target token %lld",
                              best,
                              (long long)mapped);
        }
        draft_tokens[row] = best;
        target_tokens[row] = (uint32_t)mapped;
    }

    return 0;
}

int ds4_dflash_prepare_block_inputs(const ds4_dflash_weights *w,
                                    const ds4_dflash_config *cfg,
                                    const float *tap_hc,
                                    uint32_t n_tokens,
                                    uint32_t token_index,
                                    uint32_t anchor_token,
                                    float *target_hidden,
                                    float *noise_embedding,
                                    char *err,
                                    size_t errlen) {
    if (!w || !w->loaded || !cfg || !cfg->loaded || !tap_hc ||
        !target_hidden || !noise_embedding) {
        return dflash_err(err, errlen, "invalid DFlash block input request");
    }
    if (n_tokens == 0 || token_index >= n_tokens) {
        return dflash_err(err, errlen, "DFlash block input token index is out of range");
    }
    if (anchor_token >= cfg->vocab_size || cfg->mask_token_id >= cfg->vocab_size) {
        return dflash_err(err, errlen, "DFlash block input token id is outside target vocab");
    }

    const ds4_dflash_tensor *fc = ds4_dflash_weights_find_tensor(w, "fc.weight");
    const ds4_dflash_tensor *hidden_norm = ds4_dflash_weights_find_tensor(w, "hidden_norm.weight");
    const ds4_dflash_tensor *embed = ds4_dflash_weights_find_tensor(w, "embed_tokens.weight");
    if (!fc || !hidden_norm || !embed) {
        return dflash_err(err, errlen, "DFlash block input tensors are not bound");
    }
    const uint64_t hidden = cfg->hidden_size;
    const uint64_t hc_dim = (uint64_t)cfg->hc_mult * hidden;
    const uint64_t fc_in = (uint64_t)cfg->n_target_layer_ids * hc_dim;
    if (fc->ndim != 2 || fc->shape[0] != hidden || fc->shape[1] != fc_in ||
        hidden_norm->ndim != 1 || hidden_norm->shape[0] != hidden ||
        embed->ndim != 2 || embed->shape[0] != cfg->vocab_size ||
        embed->shape[1] != hidden) {
        return dflash_err(err, errlen, "DFlash block input tensor layout does not match config");
    }

    for (uint64_t out = 0; out < hidden; out++) {
        double acc = 0.0;
        const uint64_t row = out * fc_in;
        uint64_t col = 0;
        for (uint32_t tap = 0; tap < cfg->n_target_layer_ids; tap++) {
            const float *tap_row = tap_hc + ((uint64_t)tap * n_tokens + token_index) * hc_dim;
            for (uint64_t i = 0; i < hc_dim; i++, col++) {
                acc += (double)tap_row[i] * (double)bf16_data_at(w, fc, row + col);
            }
        }
        target_hidden[out] = (float)acc;
    }

    double ss = 0.0;
    for (uint64_t i = 0; i < hidden; i++) {
        ss += (double)target_hidden[i] * (double)target_hidden[i];
    }
    const float inv_rms = 1.0f / sqrtf((float)(ss / (double)hidden) + 1.0e-6f);
    for (uint64_t i = 0; i < hidden; i++) {
        target_hidden[i] *= inv_rms * bf16_data_at(w, hidden_norm, i);
    }

    if (ds4_dflash_tensor_read_bf16_f32(w,
                                        embed,
                                        (uint64_t)anchor_token * hidden,
                                        noise_embedding,
                                        hidden,
                                        err,
                                        errlen) != 0) {
        return 1;
    }
    for (uint32_t pos = 1; pos < cfg->block_size; pos++) {
        if (ds4_dflash_tensor_read_bf16_f32(w,
                                            embed,
                                            (uint64_t)cfg->mask_token_id * hidden,
                                            noise_embedding + (uint64_t)pos * hidden,
                                            hidden,
                                            err,
                                            errlen) != 0) {
            return 1;
        }
    }
    return 0;
}
