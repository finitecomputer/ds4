#include "ds4_dflash.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define DS4_DFLASH_RMS_EPS 1.0e-6f
#define DS4_DFLASH_DEFAULT_ROPE_THETA 1000000.0f
#define DS4_DSPARK_STACK_ROUTED_EXPERTS 512u

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

static double dflash_now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

#define DS4_DSPARK_MARKOV_CACHE_MAX 64u

typedef struct {
    const ds4_dspark_weights *weights;
    uint32_t prev_token_id;
    uint32_t vocab_size;
    uint64_t age;
    float *logits;
} ds4_dspark_markov_cache_entry;

static pthread_mutex_t g_dspark_markov_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static ds4_dspark_markov_cache_entry g_dspark_markov_cache[DS4_DSPARK_MARKOV_CACHE_MAX];
static uint64_t g_dspark_markov_cache_age;

static uint32_t dspark_markov_cache_cap(void) {
    const char *env = getenv("DS4_DSPARK_MARKOV_CACHE_SIZE");
    char *end = NULL;
    unsigned long v = 32ul;
    if (getenv("DS4_DSPARK_MARKOV_CACHE_DISABLE") != NULL) return 0;
    if (env && env[0]) {
        errno = 0;
        v = strtoul(env, &end, 10);
        if (end == env || *end != '\0' || errno != 0) v = 32ul;
    }
    if (v > DS4_DSPARK_MARKOV_CACHE_MAX) v = DS4_DSPARK_MARKOV_CACHE_MAX;
    return (uint32_t)v;
}

static bool dspark_markov_cache_log_enabled(void) {
    const char *env = getenv("DS4_DSPARK_MARKOV_CACHE_LOG");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void dspark_markov_cache_forget(const ds4_dspark_weights *weights) {
    if (!weights) return;
    pthread_mutex_lock(&g_dspark_markov_cache_mutex);
    for (uint32_t i = 0; i < DS4_DSPARK_MARKOV_CACHE_MAX; i++) {
        ds4_dspark_markov_cache_entry *e = &g_dspark_markov_cache[i];
        if (e->weights == weights) {
            free(e->logits);
            memset(e, 0, sizeof(*e));
        }
    }
    pthread_mutex_unlock(&g_dspark_markov_cache_mutex);
}

static int dspark_markov_logits_cached(const ds4_dspark_weights *w,
                                       const ds4_dspark_config *cfg,
                                       uint32_t prev_token_id,
                                       const float *prev_embedding,
                                       float *logits,
                                       char *err,
                                       size_t errlen) {
    const uint32_t cap = dspark_markov_cache_cap();
    uint32_t slot = UINT32_MAX;
    uint32_t lru_slot = UINT32_MAX;
    uint64_t lru_age = UINT64_MAX;

    if (cap == 0 || !w || !cfg || !prev_embedding || !logits || cfg->vocab_size == 0) {
        return ds4_dspark_markov_logits_f32(w, cfg, prev_embedding, logits, err, errlen);
    }
    pthread_mutex_lock(&g_dspark_markov_cache_mutex);
    for (uint32_t i = 0; i < cap; i++) {
        ds4_dspark_markov_cache_entry *e = &g_dspark_markov_cache[i];
        if (e->logits &&
            e->weights == w &&
            e->prev_token_id == prev_token_id &&
            e->vocab_size == cfg->vocab_size) {
            e->age = ++g_dspark_markov_cache_age;
            memcpy(logits, e->logits, (size_t)cfg->vocab_size * sizeof(logits[0]));
            pthread_mutex_unlock(&g_dspark_markov_cache_mutex);
            if (dspark_markov_cache_log_enabled()) {
                fprintf(stderr,
                        "ds4: dspark markov cache hit token=%u vocab=%u\n",
                        prev_token_id,
                        cfg->vocab_size);
            }
            return 0;
        }
        if (!e->logits) {
            slot = i;
            break;
        }
        if (e->age < lru_age) {
            lru_age = e->age;
            lru_slot = i;
        }
    }
    if (slot == UINT32_MAX) slot = lru_slot;
    if (slot == UINT32_MAX || slot >= cap) {
        pthread_mutex_unlock(&g_dspark_markov_cache_mutex);
        return ds4_dspark_markov_logits_f32(w, cfg, prev_embedding, logits, err, errlen);
    }

    ds4_dspark_markov_cache_entry *e = &g_dspark_markov_cache[slot];
    if (!e->logits || e->vocab_size != cfg->vocab_size) {
        free(e->logits);
        e->logits = malloc((size_t)cfg->vocab_size * sizeof(e->logits[0]));
        if (!e->logits) {
            memset(e, 0, sizeof(*e));
            pthread_mutex_unlock(&g_dspark_markov_cache_mutex);
            return ds4_dspark_markov_logits_f32(w, cfg, prev_embedding, logits, err, errlen);
        }
    }
    if (ds4_dspark_markov_logits_f32(w, cfg, prev_embedding, e->logits, err, errlen) != 0) {
        e->weights = NULL;
        e->prev_token_id = 0;
        e->vocab_size = 0;
        e->age = 0;
        pthread_mutex_unlock(&g_dspark_markov_cache_mutex);
        return 1;
    }
    e->weights = w;
    e->prev_token_id = prev_token_id;
    e->vocab_size = cfg->vocab_size;
    e->age = ++g_dspark_markov_cache_age;
    memcpy(logits, e->logits, (size_t)cfg->vocab_size * sizeof(logits[0]));
    pthread_mutex_unlock(&g_dspark_markov_cache_mutex);
    if (dspark_markov_cache_log_enabled()) {
        fprintf(stderr,
                "ds4: dspark markov cache miss token=%u vocab=%u\n",
                prev_token_id,
                cfg->vocab_size);
    }
    return 0;
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

void ds4_dspark_config_init(ds4_dspark_config *cfg) {
    if (cfg) memset(cfg, 0, sizeof(*cfg));
}

void ds4_dspark_config_free(ds4_dspark_config *cfg) {
    ds4_dspark_config_init(cfg);
}

void ds4_dspark_weights_init(ds4_dspark_weights *w) {
    if (!w) return;
    memset(w, 0, sizeof(*w));
    for (uint32_t i = 0; i < DS4_DSPARK_MAX_SHARDS; i++) {
        w->shards[i].fd = -1;
    }
}

void ds4_dspark_weights_free(ds4_dspark_weights *w) {
    if (!w) return;
    dspark_markov_cache_forget(w);
    for (uint32_t i = 0; i < w->n_shards && i < DS4_DSPARK_MAX_SHARDS; i++) {
        ds4_dspark_shard *shard = &w->shards[i];
        if (shard->map && shard->file_size) munmap(shard->map, (size_t)shard->file_size);
        if (shard->fd >= 0 && (shard->loaded || shard->map)) close(shard->fd);
    }
    free(w->tensors);
    ds4_dspark_weights_init(w);
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

void ds4_dflash_hidden_history_rewind(ds4_dflash_hidden_history *h,
                                      uint32_t position_exclusive) {
    if (!h || !h->positions || h->capacity == 0) return;
    while (h->len > 0) {
        const uint32_t slot = dflash_history_slot(h, h->len - 1u);
        if (h->positions[slot] < position_exclusive) break;
        h->len--;
    }
    if (h->len == 0) h->start = 0;
}

void ds4_dflash_verify_stats_init(ds4_dflash_verify_stats *stats,
                                  uint32_t drafted,
                                  uint32_t accepted_including_anchor) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    stats->drafted = drafted;
    stats->accepted_including_anchor = accepted_including_anchor;
    stats->miss_index = -1;
    stats->miss_draft_token = -1;
    stats->miss_target_token = -1;
    stats->miss_target_top = -1;
}

bool ds4_dflash_verify_step(ds4_dflash_verify_stats *stats,
                            uint32_t index,
                            int draft_token,
                            int target_token,
                            int target_top) {
    if (!stats || stats->misses != 0 || index >= stats->drafted) return false;
    if (target_top != target_token) {
        stats->misses = 1;
        stats->rejected_draft_tokens = stats->drafted - index;
        stats->miss_index = (int)index;
        stats->miss_draft_token = draft_token;
        stats->miss_target_token = target_token;
        stats->miss_target_top = target_top;
        return false;
    }
    stats->verified++;
    stats->accepted_including_anchor++;
    return true;
}

int ds4_dflash_hidden_history_reserve(ds4_dflash_hidden_history *h,
                                      const ds4_dflash_config *cfg,
                                      uint32_t capacity,
                                      char *err,
                                      size_t errlen) {
    if (!cfg || !cfg->loaded) {
        return dflash_err(err, errlen, "invalid DFlash hidden-history reservation");
    }
    return ds4_dflash_hidden_history_reserve_raw(h,
                                                 cfg->hidden_size,
                                                 capacity,
                                                 err,
                                                 errlen);
}

int ds4_dflash_hidden_history_reserve_raw(ds4_dflash_hidden_history *h,
                                          uint32_t hidden_size,
                                          uint32_t capacity,
                                          char *err,
                                          size_t errlen) {
    float *hidden = NULL;
    uint32_t *positions = NULL;

    if (!h || hidden_size == 0 || capacity == 0) {
        return dflash_err(err, errlen, "invalid DFlash hidden-history reservation");
    }
    if ((size_t)capacity > SIZE_MAX / sizeof(hidden[0]) / hidden_size) {
        return dflash_err(err, errlen, "DFlash hidden-history reservation is too large");
    }

    hidden = calloc((size_t)capacity * hidden_size, sizeof(hidden[0]));
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
    h->hidden_size = hidden_size;
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

static const char *json_top_level_key_value(const char *json, const char *key) {
    const size_t key_len = strlen(key);
    const char *p = skip_ws(json);
    uint32_t depth = 0;
    bool in_string = false;

    if (*p != '{') return NULL;
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
            if (depth == 1 &&
                (size_t)(q - start) == key_len &&
                memcmp(start, key, key_len) == 0) {
                const char *colon = skip_ws(q + 1);
                if (*colon == ':') return colon + 1;
            }
            p = q;
            continue;
        }
        if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            if (depth == 0) return NULL;
            depth--;
            if (depth == 0) return NULL;
        }
    }
    return NULL;
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
    p = skip_ws(p);
    if (!required &&
        !strncmp(p, "null", 4) &&
        (p[4] == ',' || p[4] == '}' || isspace((unsigned char)p[4]))) {
        return 0;
    }
    return parse_u32_at(p, out, NULL, key, err, errlen);
}

static int parse_u32_key_prefer_top_level(const char *json,
                                          const char *key,
                                          uint32_t *out,
                                          bool required,
                                          char *err,
                                          size_t errlen) {
    const char *p = json_top_level_key_value(json, key);
    if (p) {
        p = skip_ws(p);
        if (!required &&
            !strncmp(p, "null", 4) &&
            (p[4] == ',' || p[4] == '}' || isspace((unsigned char)p[4]))) {
            return 0;
        }
        return parse_u32_at(p, out, NULL, key, err, errlen);
    }
    return parse_u32_key(json, key, out, required, err, errlen);
}

static int parse_f32_at(const char *p,
                        float *out,
                        const char **endp,
                        const char *name,
                        char *err,
                        size_t errlen) {
    char *end = NULL;
    float v = 0.0f;

    p = skip_ws(p);
    errno = 0;
    v = strtof(p, &end);
    if (end == p || errno == ERANGE || !isfinite(v)) {
        return dflash_err(err, errlen, "DFlash config key '%s' must be a finite number", name);
    }
    *out = v;
    if (endp) *endp = end;
    return 0;
}

static int parse_f32_key(const char *json,
                         const char *key,
                         float *out,
                         bool required,
                         char *err,
                         size_t errlen) {
    const char *p = json_key_value(json, key);
    if (!p) {
        if (!required) return 0;
        return dflash_err(err, errlen, "DFlash config is missing required key '%s'", key);
    }
    p = skip_ws(p);
    if (!required &&
        !strncmp(p, "null", 4) &&
        (p[4] == ',' || p[4] == '}' || isspace((unsigned char)p[4]))) {
        return 0;
    }
    return parse_f32_at(p, out, NULL, key, err, errlen);
}

static int parse_bool_key(const char *json,
                          const char *key,
                          bool *out,
                          bool required,
                          char *err,
                          size_t errlen) {
    const char *p = json_key_value(json, key);
    if (!p) {
        if (!required) return 0;
        return dflash_err(err, errlen, "DFlash config is missing required key '%s'", key);
    }
    p = skip_ws(p);
    if (!strncmp(p, "true", 4) &&
        (p[4] == ',' || p[4] == '}' || isspace((unsigned char)p[4]))) {
        *out = true;
        return 0;
    }
    if (!strncmp(p, "false", 5) &&
        (p[5] == ',' || p[5] == '}' || isspace((unsigned char)p[5]))) {
        *out = false;
        return 0;
    }
    if (!required &&
        !strncmp(p, "null", 4) &&
        (p[4] == ',' || p[4] == '}' || isspace((unsigned char)p[4]))) {
        return 0;
    }
    return dflash_err(err, errlen, "DFlash config key '%s' must be boolean", key);
}

static int parse_json_string_at(const char *value,
                                char *out,
                                size_t outlen,
                                const char *name,
                                char *err,
                                size_t errlen) {
    const char *p = skip_ws(value);
    size_t n = 0;

    if (!out || outlen == 0) {
        return dflash_err(err, errlen, "JSON string output is missing");
    }
    out[0] = '\0';
    if (*p != '"') {
        return dflash_err(err, errlen, "JSON key '%s' must be a string", name);
    }
    p++;
    while (*p && *p != '"') {
        if (*p == '\\') {
            return dflash_err(err,
                              errlen,
                              "JSON key '%s' uses an escaped string unsupported by this parser",
                              name);
        }
        if (n + 1u >= outlen) {
            return dflash_err(err, errlen, "JSON key '%s' string is too long", name);
        }
        out[n++] = *p++;
    }
    if (*p != '"') {
        return dflash_err(err, errlen, "JSON key '%s' has an unterminated string", name);
    }
    out[n] = '\0';
    return 0;
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

static int resolve_artifact_dir(const char *path,
                                char *out,
                                size_t outlen,
                                char *err,
                                size_t errlen) {
    struct stat st;
    int n = 0;

    if (!path || !path[0]) return dflash_err(err, errlen, "artifact path is empty");
    if (stat(path, &st) != 0) {
        return dflash_err(err, errlen, "artifact path '%s': %s", path, strerror(errno));
    }
    if (S_ISDIR(st.st_mode)) {
        n = snprintf(out, outlen, "%s", path);
    } else if (S_ISREG(st.st_mode)) {
        const char *slash = strrchr(path, '/');
        if (slash) {
            n = snprintf(out, outlen, "%.*s", (int)(slash - path), path);
        } else {
            n = snprintf(out, outlen, ".");
        }
    } else {
        return dflash_err(err, errlen, "artifact path '%s' is not a regular file or directory", path);
    }
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "artifact directory path is too long");
    }
    return 0;
}

static int join_artifact_path(const char *dir,
                              const char *name,
                              char *out,
                              size_t outlen,
                              char *err,
                              size_t errlen) {
    int n = 0;

    if (!dir || !dir[0] || !name || !name[0]) {
        return dflash_err(err, errlen, "artifact path component is empty");
    }
    n = snprintf(out, outlen, "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "artifact path is too long");
    }
    return 0;
}

static int resolve_dspark_index_path(const char *path,
                                     char *out,
                                     size_t outlen,
                                     char *err,
                                     size_t errlen) {
    struct stat st;
    char dir[DS4_DFLASH_MAX_PATH];

    if (!path || !path[0]) return dflash_err(err, errlen, "DSpark artifact path is empty");
    if (stat(path, &st) != 0) {
        return dflash_err(err, errlen, "DSpark artifact path '%s': %s", path, strerror(errno));
    }
    if (S_ISREG(st.st_mode)) {
        int n = snprintf(out, outlen, "%s", path);
        if (n < 0 || (size_t)n >= outlen) {
            return dflash_err(err, errlen, "DSpark index path is too long");
        }
        return 0;
    }
    if (!S_ISDIR(st.st_mode)) {
        return dflash_err(err, errlen, "DSpark artifact path '%s' is not a regular file or directory", path);
    }
    if (resolve_artifact_dir(path, dir, sizeof(dir), err, errlen) != 0 ||
        join_artifact_path(dir,
                           "model.safetensors.index.json",
                           out,
                           outlen,
                           err,
                           errlen) != 0) {
        return 1;
    }
    if (stat(out, &st) != 0) {
        return dflash_err(err, errlen, "DSpark index file '%s': %s", out, strerror(errno));
    }
    if (!S_ISREG(st.st_mode)) {
        return dflash_err(err, errlen, "DSpark index file '%s' is not a regular file", out);
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
        parse_u32_key_prefer_top_level(json, "sliding_window", &cfg->sliding_window, false, err, errlen) != 0 ||
        parse_u32_key(json, "max_anchors", &cfg->max_anchors, false, err, errlen) != 0 ||
        parse_f32_key(json, "rope_theta", &cfg->rope_theta, false, err, errlen) != 0 ||
        parse_bool_key(json,
                       "sliding_window_non_causal",
                       &cfg->sliding_window_non_causal,
                       false,
                       err,
                       errlen) != 0 ||
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
    if (have_aux_layer_ids) {
        /* Speculators-format DFlash configs store auxiliary hidden-state ids
         * in the same space as HF output_hidden_states, where index 0 is the
         * embedding output and decoder layer N is read at N + 1. DS4 taps
         * decoder layer outputs directly, so convert to runtime layer ids. */
        for (uint32_t i = 0; i < cfg->n_target_layer_ids; i++) {
            if (cfg->target_layer_ids[i] == 0) {
                free(json);
                ds4_dflash_config_init(cfg);
                return dflash_err(err,
                                  errlen,
                                  "DFlash aux_hidden_state_layer_ids[%u]=0 does not name a decoder layer output",
                                  i);
            }
            cfg->target_layer_ids[i]--;
        }
    }
    if (cfg->draft_vocab_size == 0) cfg->draft_vocab_size = cfg->vocab_size;
    if (cfg->target_hidden_size == 0) cfg->target_hidden_size = cfg->hidden_size;
    if (cfg->rope_theta <= 0.0f) cfg->rope_theta = DS4_DFLASH_DEFAULT_ROPE_THETA;

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
        if (i > 0 && cfg->target_layer_ids[i] <= cfg->target_layer_ids[i - 1u]) {
            return dflash_err(err, errlen,
                              "DFlash target_layer_ids must be strictly increasing");
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

int ds4_dspark_config_load(ds4_dspark_config *cfg,
                           const char *path,
                           char *err,
                           size_t errlen) {
    char resolved[DS4_DFLASH_MAX_PATH];
    char *json = NULL;

    if (!cfg) return dflash_err(err, errlen, "DSpark config output is null");
    ds4_dspark_config_init(cfg);
    if (resolve_config_path(path, resolved, sizeof(resolved), err, errlen) != 0) return 1;
    json = read_file(resolved, err, errlen);
    if (!json) return 1;

    if (parse_u32_key(json, "dspark_block_size", &cfg->block_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "dspark_noise_token_id", &cfg->noise_token_id, true, err, errlen) != 0 ||
        parse_u32_key(json, "hidden_size", &cfg->hidden_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "vocab_size", &cfg->vocab_size, true, err, errlen) != 0 ||
        parse_u32_key(json, "dspark_markov_rank", &cfg->markov_rank, true, err, errlen) != 0 ||
        parse_u32_key_prefer_top_level(json, "sliding_window", &cfg->sliding_window, true, err, errlen) != 0 ||
        parse_u32_key(json, "hc_mult", &cfg->hc_mult, true, err, errlen) != 0 ||
        parse_u32_key(json, "hc_sinkhorn_iters", &cfg->hc_sinkhorn_iters, false, err, errlen) != 0 ||
        parse_f32_key(json, "hc_eps", &cfg->hc_eps, false, err, errlen) != 0 ||
        parse_u32_key(json, "num_attention_heads", &cfg->num_attention_heads, false, err, errlen) != 0 ||
        parse_u32_key(json, "q_lora_rank", &cfg->q_lora_rank, false, err, errlen) != 0 ||
        parse_u32_key(json, "o_lora_rank", &cfg->o_lora_rank, false, err, errlen) != 0 ||
        parse_u32_key(json, "head_dim", &cfg->head_dim, false, err, errlen) != 0 ||
        parse_u32_key(json, "qk_rope_head_dim", &cfg->qk_rope_head_dim, false, err, errlen) != 0 ||
        parse_u32_key(json, "o_groups", &cfg->o_groups, false, err, errlen) != 0 ||
        parse_u32_key(json, "n_routed_experts", &cfg->n_routed_experts, false, err, errlen) != 0 ||
        parse_u32_key(json, "n_shared_experts", &cfg->n_shared_experts, false, err, errlen) != 0 ||
        parse_u32_key(json, "num_experts_per_tok", &cfg->n_activated_experts, false, err, errlen) != 0 ||
        parse_u32_key(json, "moe_intermediate_size", &cfg->moe_intermediate_size, false, err, errlen) != 0 ||
        parse_f32_key(json, "rope_theta", &cfg->rope_theta, false, err, errlen) != 0 ||
        parse_f32_key(json, "compress_rope_theta", &cfg->compress_rope_theta, false, err, errlen) != 0 ||
        parse_u32_key(json, "original_max_position_embeddings", &cfg->rope_original_max_position_embeddings, false, err, errlen) != 0 ||
        parse_f32_key(json, "factor", &cfg->rope_factor, false, err, errlen) != 0 ||
        parse_f32_key(json, "beta_fast", &cfg->rope_beta_fast, false, err, errlen) != 0 ||
        parse_f32_key(json, "beta_slow", &cfg->rope_beta_slow, false, err, errlen) != 0 ||
        parse_f32_key(json, "routed_scaling_factor", &cfg->routed_scaling_factor, false, err, errlen) != 0 ||
        parse_f32_key(json, "swiglu_limit", &cfg->swiglu_limit, false, err, errlen) != 0 ||
        parse_u32_array_key(json,
                            "dspark_target_layer_ids",
                            cfg->target_layer_ids,
                            DS4_DFLASH_MAX_TARGET_LAYERS,
                            &cfg->n_target_layer_ids,
                            err,
                            errlen) != 0) {
        free(json);
        ds4_dspark_config_init(cfg);
        return 1;
    }
    cfg->n_mtp_stages = cfg->n_target_layer_ids;
    if (cfg->hc_sinkhorn_iters == 0) cfg->hc_sinkhorn_iters = 20;
    if (cfg->hc_eps <= 0.0f) cfg->hc_eps = DS4_DFLASH_RMS_EPS;
    if (cfg->qk_rope_head_dim == 0 && cfg->head_dim != 0) {
        cfg->qk_rope_head_dim = cfg->head_dim < 64u ? cfg->head_dim : 64u;
    }
    if (cfg->rope_theta <= 0.0f) cfg->rope_theta = 10000.0f;
    if (cfg->compress_rope_theta <= 0.0f) cfg->compress_rope_theta = cfg->rope_theta;
    if (cfg->rope_factor <= 0.0f) cfg->rope_factor = 1.0f;
    if (cfg->rope_beta_fast <= 0.0f) cfg->rope_beta_fast = 32.0f;
    if (cfg->rope_beta_slow <= 0.0f) cfg->rope_beta_slow = 1.0f;
    if (cfg->routed_scaling_factor <= 0.0f) cfg->routed_scaling_factor = 1.0f;
    if (cfg->swiglu_limit < 0.0f) cfg->swiglu_limit = 0.0f;

    snprintf(cfg->source_path, sizeof(cfg->source_path), "%s", resolved);
    cfg->loaded = true;
    free(json);
    return 0;
}

int ds4_dspark_config_validate_target(const ds4_dspark_config *cfg,
                                      uint32_t target_hidden_size,
                                      uint32_t target_vocab_size,
                                      uint32_t target_n_layer,
                                      char *err,
                                      size_t errlen) {
    if (!cfg || !cfg->loaded) {
        return dflash_err(err, errlen, "DSpark config is not loaded");
    }
    if (cfg->block_size < 2 || cfg->block_size > 16) {
        return dflash_err(err, errlen,
                          "DSpark dspark_block_size %u is outside the supported range 2..16",
                          cfg->block_size);
    }
    if (cfg->hidden_size != target_hidden_size) {
        return dflash_err(err, errlen,
                          "DSpark hidden_size %u does not match target hidden_size %u",
                          cfg->hidden_size, target_hidden_size);
    }
    if (cfg->vocab_size != target_vocab_size) {
        return dflash_err(err, errlen,
                          "DSpark vocab_size %u does not match target vocab_size %u",
                          cfg->vocab_size, target_vocab_size);
    }
    if (cfg->noise_token_id >= cfg->vocab_size) {
        return dflash_err(err, errlen,
                          "DSpark dspark_noise_token_id %u is outside vocab_size %u",
                          cfg->noise_token_id, cfg->vocab_size);
    }
    if (cfg->markov_rank == 0 || cfg->markov_rank > 4096) {
        return dflash_err(err, errlen,
                          "DSpark dspark_markov_rank %u is outside the supported range",
                          cfg->markov_rank);
    }
    if (cfg->hc_mult == 0 || cfg->hc_mult > 16) {
        return dflash_err(err, errlen, "DSpark hc_mult %u is invalid", cfg->hc_mult);
    }
    if (cfg->sliding_window == 0) {
        return dflash_err(err, errlen, "DSpark sliding_window must be non-zero");
    }
    if (cfg->n_target_layer_ids == 0) {
        return dflash_err(err, errlen, "DSpark dspark_target_layer_ids must not be empty");
    }
    for (uint32_t i = 0; i < cfg->n_target_layer_ids; i++) {
        if (cfg->target_layer_ids[i] >= target_n_layer) {
            return dflash_err(err, errlen,
                              "DSpark dspark_target_layer_ids[%u]=%u is outside target layer count %u",
                              i, cfg->target_layer_ids[i], target_n_layer);
        }
        if (i > 0 && cfg->target_layer_ids[i] <= cfg->target_layer_ids[i - 1u]) {
            return dflash_err(err, errlen,
                              "DSpark dspark_target_layer_ids must be strictly increasing");
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
    } else if (len == 3 && memcmp(p, "F32", 3) == 0) {
        *dtype = DS4_DFLASH_TENSOR_F32;
    } else if (len == 7 && memcmp(p, "F8_E4M3", 7) == 0) {
        *dtype = DS4_DFLASH_TENSOR_F8_E4M3;
    } else if (len == 7 && memcmp(p, "F8_E8M0", 7) == 0) {
        *dtype = DS4_DFLASH_TENSOR_F8_E8M0;
    } else if (len == 2 && memcmp(p, "I8", 2) == 0) {
        *dtype = DS4_DFLASH_TENSOR_I8;
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
        case DS4_DFLASH_TENSOR_F32: return "F32";
        case DS4_DFLASH_TENSOR_F8_E4M3: return "F8_E4M3";
        case DS4_DFLASH_TENSOR_F8_E8M0: return "F8_E8M0";
        case DS4_DFLASH_TENSOR_I8: return "I8";
        case DS4_DFLASH_TENSOR_BOOL: return "BOOL";
        case DS4_DFLASH_TENSOR_I64: return "I64";
        default: return "unknown";
    }
}

static uint64_t dtype_size(ds4_dflash_tensor_dtype dtype) {
    switch (dtype) {
        case DS4_DFLASH_TENSOR_BF16: return 2;
        case DS4_DFLASH_TENSOR_F32: return 4;
        case DS4_DFLASH_TENSOR_F8_E4M3:
        case DS4_DFLASH_TENSOR_F8_E8M0:
        case DS4_DFLASH_TENSOR_I8:
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

static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
    return b ? (a + b - 1u) / b : 0;
}

static int dspark_index_weight_file(const char *index_json,
                                    const char *name,
                                    char *file_name,
                                    size_t file_name_len,
                                    char *err,
                                    size_t errlen) {
    const char *weight_map_value = json_key_value(index_json, "weight_map");
    const char *weight_map_start = NULL;
    const char *weight_map_end = NULL;
    const char *value = NULL;

    if (!weight_map_value ||
        json_object_span(weight_map_value, &weight_map_start, &weight_map_end) != 0) {
        return dflash_err(err, errlen, "DSpark index is missing object key 'weight_map'");
    }
    value = json_key_value_bounded(weight_map_start, weight_map_end, name);
    if (!value) {
        return dflash_err(err, errlen, "DSpark index is missing tensor '%s'", name);
    }
    return parse_json_string_at(value, file_name, file_name_len, name, err, errlen);
}

static int dspark_find_or_open_shard(ds4_dspark_weights *w,
                                     const char *artifact_dir,
                                     const char *file_name,
                                     char **headers,
                                     uint32_t *out_index,
                                     char *err,
                                     size_t errlen) {
    char resolved[DS4_DFLASH_MAX_PATH];
    char *header = NULL;
    uint64_t header_len = 0;
    struct stat st;
    int fd = -1;
    void *map = NULL;

    if (strncmp(file_name, "dspark-mtp-", 11) != 0 ||
        !str_ends_with(file_name, ".safetensors")) {
        return dflash_err(err,
                          errlen,
                          "DSpark tensor is mapped to non-MTP shard '%s'",
                          file_name);
    }
    if (join_artifact_path(artifact_dir,
                           file_name,
                           resolved,
                           sizeof(resolved),
                           err,
                           errlen) != 0) {
        return 1;
    }
    for (uint32_t i = 0; i < w->n_shards; i++) {
        if (strcmp(w->shards[i].source_path, resolved) == 0) {
            *out_index = i;
            return 0;
        }
    }
    if (w->n_shards >= DS4_DSPARK_MAX_SHARDS) {
        return dflash_err(err, errlen, "DSpark artifact has more than %u MTP shards", DS4_DSPARK_MAX_SHARDS);
    }

    header = read_safetensors_header(resolved, &header_len, err, errlen);
    if (!header) return 1;
    fd = open(resolved, O_RDONLY);
    if (fd < 0) {
        free(header);
        return dflash_err(err, errlen, "DSpark weights file '%s': %s", resolved, strerror(errno));
    }
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DSpark weights file '%s': stat failed", resolved);
    }
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DSpark weights file '%s' is too large to map", resolved);
    }
    if ((uint64_t)st.st_size < 8u + header_len) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DSpark weights file '%s' is shorter than its header", resolved);
    }
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        free(header);
        return dflash_err(err, errlen, "DSpark weights file '%s': mmap failed: %s", resolved, strerror(errno));
    }

    *out_index = w->n_shards++;
    snprintf(w->shards[*out_index].source_path,
             sizeof(w->shards[*out_index].source_path),
             "%s",
             resolved);
    w->shards[*out_index].header_len = header_len;
    w->shards[*out_index].file_size = (uint64_t)st.st_size;
    w->shards[*out_index].data_start = 8u + header_len;
    w->shards[*out_index].map = map;
    w->shards[*out_index].fd = fd;
    w->shards[*out_index].loaded = true;
    headers[*out_index] = header;
    return 0;
}

static int bind_dspark_tensor(ds4_dspark_weights *w,
                              const char *header,
                              uint32_t shard_index,
                              const char *name,
                              ds4_dflash_tensor_dtype dtype,
                              const uint64_t *shape,
                              uint32_t ndim,
                              char *err,
                              size_t errlen) {
    dflash_tensor_meta meta;
    uint64_t expected_bytes = 0;
    ds4_dflash_tensor *t = NULL;
    ds4_dspark_shard *shard = NULL;
    int n = 0;

    if (!w || !w->tensors || w->n_bound_tensors >= w->n_tensors) {
        return dflash_err(err, errlen, "DSpark safetensors tensor table is full");
    }
    if (shard_index >= w->n_shards) {
        return dflash_err(err, errlen, "DSpark safetensors tensor '%s' has invalid shard", name);
    }
    if (expect_tensor(header, name, dtype, shape, ndim, err, errlen) != 0 ||
        read_tensor_meta(header, name, &meta, err, errlen) != 0 ||
        tensor_expected_bytes(dtype, shape, ndim, &expected_bytes, err, errlen) != 0) {
        return 1;
    }
    if (meta.data_offsets[1] - meta.data_offsets[0] != expected_bytes) {
        return dflash_err(err, errlen,
                          "DSpark safetensors tensor '%s' has %llu bytes, expected %llu",
                          name,
                          (unsigned long long)(meta.data_offsets[1] - meta.data_offsets[0]),
                          (unsigned long long)expected_bytes);
    }

    shard = &w->shards[shard_index];
    if (shard->data_start > UINT64_MAX - meta.data_offsets[0] ||
        shard->data_start + meta.data_offsets[1] > shard->file_size) {
        return dflash_err(err, errlen,
                          "DSpark safetensors tensor '%s' extends past mapped shard",
                          name);
    }

    t = &w->tensors[w->n_bound_tensors++];
    memset(t, 0, sizeof(*t));
    n = snprintf(t->name, sizeof(t->name), "%s", name);
    if (n < 0 || (size_t)n >= sizeof(t->name)) {
        return dflash_err(err, errlen, "DSpark safetensors tensor name is too long");
    }
    t->dtype = dtype;
    t->ndim = ndim;
    memcpy(t->shape, shape, (size_t)ndim * sizeof(t->shape[0]));
    t->data_offsets[0] = meta.data_offsets[0];
    t->data_offsets[1] = meta.data_offsets[1];
    t->abs_offset = shard->data_start + meta.data_offsets[0];
    t->nbytes = expected_bytes;
    t->shard_index = shard_index;
    return 0;
}

static int visit_dspark_tensor(ds4_dspark_weights *w,
                               const char *artifact_dir,
                               const char *index_json,
                               char **headers,
                               const char *name,
                               ds4_dflash_tensor_dtype dtype,
                               const uint64_t *shape,
                               uint32_t ndim,
                               char *err,
                               size_t errlen) {
    char file_name[DS4_DFLASH_MAX_PATH];
    uint32_t shard_index = 0;

    if (dspark_index_weight_file(index_json,
                                 name,
                                 file_name,
                                 sizeof(file_name),
                                 err,
                                 errlen) != 0 ||
        dspark_find_or_open_shard(w,
                                  artifact_dir,
                                  file_name,
                                  headers,
                                  &shard_index,
                                  err,
                                  errlen) != 0) {
        return 1;
    }
    return bind_dspark_tensor(w,
                              headers[shard_index],
                              shard_index,
                              name,
                              dtype,
                              shape,
                              ndim,
                              err,
                              errlen);
}

static uint32_t dspark_required_tensor_count(void) {
    return 10u;
}

static uint32_t dspark_graph_tensor_count(const ds4_dspark_config *cfg) {
    if (!cfg) return 0;
    return dspark_required_tensor_count() +
           cfg->n_mtp_stages * (29u + cfg->n_routed_experts * 6u);
}

static int dspark_stage_tensor_name(char *out,
                                    size_t outlen,
                                    uint32_t stage,
                                    const char *suffix,
                                    char *err,
                                    size_t errlen) {
    int n = snprintf(out, outlen, "mtp.%u.%s", stage, suffix);
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "DSpark stage tensor name is too long");
    }
    return 0;
}

static int dspark_visit_stage_tensor(ds4_dspark_weights *w,
                                     const char *artifact_dir,
                                     const char *index_json,
                                     char **headers,
                                     uint32_t stage,
                                     const char *suffix,
                                     ds4_dflash_tensor_dtype dtype,
                                     const uint64_t *shape,
                                     uint32_t ndim,
                                     char *err,
                                     size_t errlen) {
    char name[DS4_DFLASH_MAX_TENSOR_NAME];
    if (dspark_stage_tensor_name(name, sizeof(name), stage, suffix, err, errlen) != 0) {
        return 1;
    }
    return visit_dspark_tensor(w,
                               artifact_dir,
                               index_json,
                               headers,
                               name,
                               dtype,
                               shape,
                               ndim,
                               err,
                               errlen);
}

static int dspark_visit_stage_graph_tensors(ds4_dspark_weights *w,
                                            const char *artifact_dir,
                                            const char *index_json,
                                            char **headers,
                                            const ds4_dspark_config *cfg,
                                            uint32_t stage,
                                            char *err,
                                            size_t errlen) {
    const uint64_t hidden = cfg->hidden_size;
    const uint64_t hc_mult = cfg->hc_mult;
    const uint64_t mix_hc = (2u + hc_mult) * hc_mult;
    const uint64_t hc_dim = hc_mult * hidden;
    const uint64_t heads = cfg->num_attention_heads;
    const uint64_t head_dim = cfg->head_dim;
    const uint64_t q_lora = cfg->q_lora_rank;
    const uint64_t q_dim = heads * head_dim;
    const uint64_t o_groups = cfg->o_groups;
    const uint64_t o_lora = cfg->o_lora_rank;
    const uint64_t o_lora_dim = o_groups * o_lora;
    const uint64_t wo_a_in = q_dim / o_groups;
    const uint64_t n_experts = cfg->n_routed_experts;
    const uint64_t moe = cfg->moe_intermediate_size;

    const uint64_t hc_fn_shape[] = {mix_hc, hc_dim};
    const uint64_t hc_base_shape[] = {mix_hc};
    const uint64_t hc_scale_shape[] = {3};
    const uint64_t hidden_shape[] = {hidden};
    const uint64_t q_lora_shape[] = {q_lora};
    const uint64_t head_shape[] = {head_dim};
    const uint64_t attn_sink_shape[] = {heads};
    const uint64_t wq_a_shape[] = {q_lora, hidden};
    const uint64_t wq_a_scale_shape[] = {ceil_div_u64(q_lora, 128u), ceil_div_u64(hidden, 128u)};
    const uint64_t wq_b_shape[] = {q_dim, q_lora};
    const uint64_t wq_b_scale_shape[] = {ceil_div_u64(q_dim, 128u), ceil_div_u64(q_lora, 128u)};
    const uint64_t wkv_shape[] = {head_dim, hidden};
    const uint64_t wkv_scale_shape[] = {ceil_div_u64(head_dim, 128u), ceil_div_u64(hidden, 128u)};
    const uint64_t wo_a_shape[] = {o_lora_dim, wo_a_in};
    const uint64_t wo_a_scale_shape[] = {ceil_div_u64(o_lora_dim, 128u), ceil_div_u64(wo_a_in, 128u)};
    const uint64_t wo_b_shape[] = {hidden, o_lora_dim};
    const uint64_t wo_b_scale_shape[] = {ceil_div_u64(hidden, 128u), ceil_div_u64(o_lora_dim, 128u)};
    const uint64_t gate_weight_shape[] = {n_experts, hidden};
    const uint64_t gate_bias_shape[] = {n_experts};
    const uint64_t shared_w1_shape[] = {moe, hidden};
    const uint64_t shared_w1_scale_shape[] = {ceil_div_u64(moe, 128u), ceil_div_u64(hidden, 128u)};
    const uint64_t shared_w2_shape[] = {hidden, moe};
    const uint64_t shared_w2_scale_shape[] = {ceil_div_u64(hidden, 128u), ceil_div_u64(moe, 128u)};
    const uint64_t expert_w1_shape[] = {moe, hidden / 2u};
    const uint64_t expert_w1_scale_shape[] = {moe, ceil_div_u64(hidden, 32u)};
    const uint64_t expert_w2_shape[] = {hidden, moe / 2u};
    const uint64_t expert_w2_scale_shape[] = {hidden, ceil_div_u64(moe, 32u)};

    if (hidden == 0 || hc_mult == 0 || heads == 0 || head_dim == 0 ||
        q_lora == 0 || o_groups == 0 || o_lora == 0 || n_experts == 0 ||
        moe == 0 || q_dim % o_groups != 0 || hidden % 2u != 0 || moe % 2u != 0) {
        return dflash_err(err,
                          errlen,
                          "DSpark graph dimensions are incomplete or unsupported");
    }
    if (hc_mult > UINT64_MAX / hidden ||
        (2u + hc_mult) > UINT64_MAX / hc_mult ||
        heads > UINT64_MAX / head_dim ||
        o_groups > UINT64_MAX / o_lora) {
        return dflash_err(err, errlen, "DSpark graph tensor shape overflows");
    }

    if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "hc_attn_fn", DS4_DFLASH_TENSOR_F32,
                                  hc_fn_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "hc_attn_base", DS4_DFLASH_TENSOR_F32,
                                  hc_base_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "hc_attn_scale", DS4_DFLASH_TENSOR_F32,
                                  hc_scale_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "hc_ffn_fn", DS4_DFLASH_TENSOR_F32,
                                  hc_fn_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "hc_ffn_base", DS4_DFLASH_TENSOR_F32,
                                  hc_base_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "hc_ffn_scale", DS4_DFLASH_TENSOR_F32,
                                  hc_scale_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wq_a.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  wq_a_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wq_a.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  wq_a_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.q_norm.weight", DS4_DFLASH_TENSOR_BF16,
                                  q_lora_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wq_b.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  wq_b_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wq_b.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  wq_b_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wkv.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  wkv_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wkv.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  wkv_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.kv_norm.weight", DS4_DFLASH_TENSOR_BF16,
                                  head_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wo_a.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  wo_a_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wo_a.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  wo_a_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wo_b.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  wo_b_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.wo_b.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  wo_b_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn.attn_sink", DS4_DFLASH_TENSOR_F32,
                                  attn_sink_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "attn_norm.weight", DS4_DFLASH_TENSOR_BF16,
                                  hidden_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn_norm.weight", DS4_DFLASH_TENSOR_BF16,
                                  hidden_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.gate.weight", DS4_DFLASH_TENSOR_BF16,
                                  gate_weight_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.gate.bias", DS4_DFLASH_TENSOR_F32,
                                  gate_bias_shape, 1, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.shared_experts.w1.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  shared_w1_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.shared_experts.w1.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  shared_w1_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.shared_experts.w2.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  shared_w2_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.shared_experts.w2.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  shared_w2_scale_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.shared_experts.w3.weight", DS4_DFLASH_TENSOR_F8_E4M3,
                                  shared_w1_shape, 2, err, errlen) != 0 ||
        dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                  "ffn.shared_experts.w3.scale", DS4_DFLASH_TENSOR_F8_E8M0,
                                  shared_w1_scale_shape, 2, err, errlen) != 0) {
        return 1;
    }

    for (uint32_t expert = 0; expert < cfg->n_routed_experts; expert++) {
        char suffix[DS4_DFLASH_MAX_TENSOR_NAME];
        int n = snprintf(suffix, sizeof(suffix), "ffn.experts.%u.w1.weight", expert);
        if (n < 0 || (size_t)n >= sizeof(suffix)) return dflash_err(err, errlen, "DSpark expert tensor name is too long");
        if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                      suffix, DS4_DFLASH_TENSOR_I8,
                                      expert_w1_shape, 2, err, errlen) != 0) return 1;
        n = snprintf(suffix, sizeof(suffix), "ffn.experts.%u.w1.scale", expert);
        if (n < 0 || (size_t)n >= sizeof(suffix)) return dflash_err(err, errlen, "DSpark expert tensor name is too long");
        if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                      suffix, DS4_DFLASH_TENSOR_F8_E8M0,
                                      expert_w1_scale_shape, 2, err, errlen) != 0) return 1;
        n = snprintf(suffix, sizeof(suffix), "ffn.experts.%u.w2.weight", expert);
        if (n < 0 || (size_t)n >= sizeof(suffix)) return dflash_err(err, errlen, "DSpark expert tensor name is too long");
        if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                      suffix, DS4_DFLASH_TENSOR_I8,
                                      expert_w2_shape, 2, err, errlen) != 0) return 1;
        n = snprintf(suffix, sizeof(suffix), "ffn.experts.%u.w2.scale", expert);
        if (n < 0 || (size_t)n >= sizeof(suffix)) return dflash_err(err, errlen, "DSpark expert tensor name is too long");
        if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                      suffix, DS4_DFLASH_TENSOR_F8_E8M0,
                                      expert_w2_scale_shape, 2, err, errlen) != 0) return 1;
        n = snprintf(suffix, sizeof(suffix), "ffn.experts.%u.w3.weight", expert);
        if (n < 0 || (size_t)n >= sizeof(suffix)) return dflash_err(err, errlen, "DSpark expert tensor name is too long");
        if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                      suffix, DS4_DFLASH_TENSOR_I8,
                                      expert_w1_shape, 2, err, errlen) != 0) return 1;
        n = snprintf(suffix, sizeof(suffix), "ffn.experts.%u.w3.scale", expert);
        if (n < 0 || (size_t)n >= sizeof(suffix)) return dflash_err(err, errlen, "DSpark expert tensor name is too long");
        if (dspark_visit_stage_tensor(w, artifact_dir, index_json, headers, stage,
                                      suffix, DS4_DFLASH_TENSOR_F8_E8M0,
                                      expert_w1_scale_shape, 2, err, errlen) != 0) return 1;
    }

    return 0;
}

static int visit_dspark_required_tensors(ds4_dspark_weights *w,
                                         const char *artifact_dir,
                                         const char *index_json,
                                         char **headers,
                                         const ds4_dspark_config *cfg,
                                         char *err,
                                         size_t errlen) {
    const uint64_t hidden = cfg->hidden_size;
    const uint64_t target_layers = cfg->n_target_layer_ids;
    const uint64_t markov_rank = cfg->markov_rank;
    const uint64_t vocab = cfg->vocab_size;
    const uint64_t hc_mult = cfg->hc_mult;
    uint64_t main_in = 0;
    uint64_t hc_dim = 0;
    uint64_t confidence_in = 0;

    if (target_layers == 0 || hidden == 0 || markov_rank == 0 || hc_mult == 0) {
        return dflash_err(err, errlen, "DSpark config is missing required dimensions for weights open");
    }
    if (hidden > UINT64_MAX / target_layers || hc_mult > UINT64_MAX / hidden ||
        hidden > UINT64_MAX - markov_rank) {
        return dflash_err(err, errlen, "DSpark required tensor shape overflows");
    }
    main_in = hidden * target_layers;
    hc_dim = hc_mult * hidden;
    confidence_in = hidden + markov_rank;

    const uint64_t main_proj_shape[] = {hidden, main_in};
    const uint64_t main_scale_shape[] = {ceil_div_u64(hidden, 128u),
                                         ceil_div_u64(main_in, 128u)};
    const uint64_t hidden_shape[] = {hidden};
    const uint64_t markov_shape[] = {vocab, markov_rank};
    const uint64_t confidence_shape[] = {1, confidence_in};
    const uint64_t hc_fn_shape[] = {hc_mult, hc_dim};
    const uint64_t hc_base_shape[] = {hc_mult};
    const uint64_t hc_scale_shape[] = {1};

    if (visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.0.main_proj.weight",
                            DS4_DFLASH_TENSOR_F8_E4M3,
                            main_proj_shape, 2, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.0.main_proj.scale",
                            DS4_DFLASH_TENSOR_F8_E8M0,
                            main_scale_shape, 2, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.0.main_norm.weight",
                            DS4_DFLASH_TENSOR_BF16,
                            hidden_shape, 1, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.norm.weight",
                            DS4_DFLASH_TENSOR_BF16,
                            hidden_shape, 1, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.markov_head.markov_w1.weight",
                            DS4_DFLASH_TENSOR_BF16,
                            markov_shape, 2, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.markov_head.markov_w2.weight",
                            DS4_DFLASH_TENSOR_BF16,
                            markov_shape, 2, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.confidence_head.proj.weight",
                            DS4_DFLASH_TENSOR_BF16,
                            confidence_shape, 2, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.hc_head_fn",
                            DS4_DFLASH_TENSOR_F32,
                            hc_fn_shape, 2, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.hc_head_base",
                            DS4_DFLASH_TENSOR_F32,
                            hc_base_shape, 1, err, errlen) != 0 ||
        visit_dspark_tensor(w, artifact_dir, index_json, headers,
                            "mtp.2.hc_head_scale",
                            DS4_DFLASH_TENSOR_F32,
                            hc_scale_shape, 1, err, errlen) != 0) {
        return 1;
    }
    return 0;
}

static int dspark_weights_open_impl(ds4_dspark_weights *w,
                                    const char *path,
                                    const ds4_dspark_config *cfg,
                                    bool bind_graph,
                                    char *err,
                                    size_t errlen) {
    char artifact_dir[DS4_DFLASH_MAX_PATH];
    char index_path[DS4_DFLASH_MAX_PATH];
    char *index_json = NULL;
    char *headers[DS4_DSPARK_MAX_SHARDS] = {0};
    uint32_t n_required = bind_graph ?
        dspark_graph_tensor_count(cfg) :
        dspark_required_tensor_count();

    if (!w) return dflash_err(err, errlen, "DSpark weights output is null");
    ds4_dspark_weights_init(w);
    if (!cfg || !cfg->loaded) {
        return dflash_err(err, errlen, "DSpark config must be loaded before weights open");
    }
    if (bind_graph &&
        (cfg->n_mtp_stages == 0 ||
         cfg->num_attention_heads == 0 ||
         cfg->q_lora_rank == 0 ||
         cfg->o_lora_rank == 0 ||
         cfg->head_dim == 0 ||
         cfg->o_groups == 0 ||
         cfg->n_routed_experts == 0 ||
         cfg->n_shared_experts != 1 ||
         cfg->n_activated_experts == 0 ||
         cfg->moe_intermediate_size == 0)) {
        return dflash_err(err, errlen, "DSpark config is missing graph dimensions required for weights open");
    }
    if (resolve_artifact_dir(path, artifact_dir, sizeof(artifact_dir), err, errlen) != 0 ||
        resolve_dspark_index_path(path, index_path, sizeof(index_path), err, errlen) != 0) {
        return 1;
    }
    index_json = read_file(index_path, err, errlen);
    if (!index_json) return 1;

    w->tensors = calloc(n_required, sizeof(w->tensors[0]));
    if (!w->tensors) {
        free(index_json);
        return dflash_err(err, errlen, "out of memory binding DSpark safetensors");
    }
    w->n_tensors = n_required;
    snprintf(w->source_path, sizeof(w->source_path), "%s", artifact_dir);
    snprintf(w->index_path, sizeof(w->index_path), "%s", index_path);

    if (visit_dspark_required_tensors(w,
                                      artifact_dir,
                                      index_json,
                                      headers,
                                      cfg,
                                      err,
                                      errlen) != 0) {
        for (uint32_t i = 0; i < DS4_DSPARK_MAX_SHARDS; i++) free(headers[i]);
        free(index_json);
        ds4_dspark_weights_free(w);
        return 1;
    }
    if (bind_graph) {
        for (uint32_t stage = 0; stage < cfg->n_mtp_stages; stage++) {
            if (dspark_visit_stage_graph_tensors(w,
                                                 artifact_dir,
                                                 index_json,
                                                 headers,
                                                 cfg,
                                                 stage,
                                                 err,
                                                 errlen) != 0) {
                for (uint32_t i = 0; i < DS4_DSPARK_MAX_SHARDS; i++) free(headers[i]);
                free(index_json);
                ds4_dspark_weights_free(w);
                return 1;
            }
        }
    }

    w->loaded = true;
    for (uint32_t i = 0; i < DS4_DSPARK_MAX_SHARDS; i++) free(headers[i]);
    free(index_json);
    return 0;
}

int ds4_dspark_weights_open(ds4_dspark_weights *w,
                            const char *path,
                            const ds4_dspark_config *cfg,
                            char *err,
                            size_t errlen) {
    return dspark_weights_open_impl(w, path, cfg, false, err, errlen);
}

int ds4_dspark_weights_open_graph(ds4_dspark_weights *w,
                                  const char *path,
                                  const ds4_dspark_config *cfg,
                                  char *err,
                                  size_t errlen) {
    return dspark_weights_open_impl(w, path, cfg, true, err, errlen);
}

int ds4_dspark_weights_validate(ds4_dspark_weights *w,
                                const char *path,
                                const ds4_dspark_config *cfg,
                                char *err,
                                size_t errlen) {
    return ds4_dspark_weights_open(w, path, cfg, err, errlen);
}

const ds4_dflash_tensor *ds4_dspark_weights_find_tensor(const ds4_dspark_weights *w,
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

static float dspark_decode_fp8_e4m3_slow(uint8_t raw) {
    const int sign = (raw & 0x80u) ? -1 : 1;
    const int exp = (int)((raw >> 3u) & 0x0fu);
    const int mant = (int)(raw & 0x07u);
    float v = 0.0f;

    if (exp == 0) {
        v = mant ? ldexpf((float)mant / 8.0f, -6) : 0.0f;
    } else if (exp == 15 && mant == 7) {
        v = 448.0f;
    } else {
        v = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    return sign < 0 ? -v : v;
}

static float dspark_decode_fp8_e8m0_slow(uint8_t raw) {
    if (raw == 0xffu) return FLT_MAX;
    return ldexpf(1.0f, (int)raw - 127);
}

static float dspark_decode_fp4_e2m1_slow(uint8_t raw) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float v = values[raw & 0x07u];
    return (raw & 0x08u) ? -v : v;
}

static pthread_once_t dspark_decode_tables_once = PTHREAD_ONCE_INIT;
static float dspark_bf16_table[65536];
static float dspark_fp8_e4m3_table[256];
static float dspark_fp8_e8m0_table[256];
static float dspark_fp4_e2m1_table[16];

static void dspark_decode_tables_init_once(void) {
    for (uint32_t i = 0; i < 65536u; i++) {
        dspark_bf16_table[i] = bf16_to_f32((uint16_t)i);
    }
    for (uint32_t i = 0; i < 256u; i++) {
        dspark_fp8_e4m3_table[i] = dspark_decode_fp8_e4m3_slow((uint8_t)i);
        dspark_fp8_e8m0_table[i] = dspark_decode_fp8_e8m0_slow((uint8_t)i);
    }
    for (uint32_t i = 0; i < 16u; i++) {
        dspark_fp4_e2m1_table[i] = dspark_decode_fp4_e2m1_slow((uint8_t)i);
    }
}

static void dspark_decode_tables_ensure(void) {
    (void)pthread_once(&dspark_decode_tables_once, dspark_decode_tables_init_once);
}

#define DS4_DSPARK_MAX_PARALLEL_THREADS 64u
#define DS4_DSPARK_PARALLEL_MIN_OPS 262144u

typedef void (*dspark_parallel_worker_fn)(void *tasks, uint32_t index);

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    pthread_t threads[DS4_DSPARK_MAX_PARALLEL_THREADS];
    uint32_t indices[DS4_DSPARK_MAX_PARALLEL_THREADS];
    uint32_t n_threads;
    uint32_t n_tasks;
    uint32_t done_count;
    uint64_t generation;
    dspark_parallel_worker_fn fn;
    void *tasks;
    bool initialized;
    bool stop;
} dspark_parallel_pool;

static dspark_parallel_pool g_dspark_pool;

static uint32_t dspark_online_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) return 1;
    if (n > (long)DS4_DSPARK_MAX_PARALLEL_THREADS) return DS4_DSPARK_MAX_PARALLEL_THREADS;
    return (uint32_t)n;
}

static uint32_t dspark_requested_threads(void) {
    const char *env = getenv("DS4_DSPARK_THREADS");
    char *end = NULL;
    unsigned long v = 0;

    if (!env || !*env) return 1;
    if (strcmp(env, "auto") == 0) return dspark_online_cpu_count();

    errno = 0;
    v = strtoul(env, &end, 10);
    if (end == env || errno == ERANGE) return 1;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end) return 1;
    if (v < 1) return 1;
    if (v > DS4_DSPARK_MAX_PARALLEL_THREADS) return DS4_DSPARK_MAX_PARALLEL_THREADS;
    return (uint32_t)v;
}

static uint32_t dspark_parallel_threads_for(uint64_t rows, uint64_t cols) {
    uint32_t threads = dspark_requested_threads();
    if (threads <= 1 || rows < 2 || cols == 0) return 1;
    if (rows < ceil_div_u64(DS4_DSPARK_PARALLEL_MIN_OPS, cols)) return 1;
    if (rows < (uint64_t)threads) threads = (uint32_t)rows;
    return threads;
}

static void *dspark_parallel_pool_worker_main(void *arg) {
    const uint32_t index = (uint32_t)(uintptr_t)arg;
    uint64_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_dspark_pool.mutex);
        while (!g_dspark_pool.stop &&
               seen_generation == g_dspark_pool.generation) {
            pthread_cond_wait(&g_dspark_pool.start_cond, &g_dspark_pool.mutex);
        }
        if (g_dspark_pool.stop) {
            pthread_mutex_unlock(&g_dspark_pool.mutex);
            return NULL;
        }
        const uint64_t generation = g_dspark_pool.generation;
        dspark_parallel_worker_fn fn = g_dspark_pool.fn;
        void *tasks = g_dspark_pool.tasks;
        const uint32_t n_tasks = g_dspark_pool.n_tasks;
        pthread_mutex_unlock(&g_dspark_pool.mutex);

        if (fn && index < n_tasks) fn(tasks, index);

        pthread_mutex_lock(&g_dspark_pool.mutex);
        seen_generation = generation;
        g_dspark_pool.done_count++;
        if (g_dspark_pool.done_count >= g_dspark_pool.n_threads) {
            pthread_cond_signal(&g_dspark_pool.done_cond);
        }
        pthread_mutex_unlock(&g_dspark_pool.mutex);
    }
}

static bool dspark_parallel_pool_ensure(uint32_t n_threads) {
    if (n_threads <= 1) return true;
    if (n_threads > DS4_DSPARK_MAX_PARALLEL_THREADS) {
        n_threads = DS4_DSPARK_MAX_PARALLEL_THREADS;
    }
    if (g_dspark_pool.initialized) return g_dspark_pool.n_threads >= n_threads;

    memset(&g_dspark_pool, 0, sizeof(g_dspark_pool));
    pthread_mutex_init(&g_dspark_pool.mutex, NULL);
    pthread_cond_init(&g_dspark_pool.start_cond, NULL);
    pthread_cond_init(&g_dspark_pool.done_cond, NULL);
    g_dspark_pool.n_threads = n_threads;
    for (uint32_t i = 0; i < n_threads; i++) {
        g_dspark_pool.indices[i] = i;
        if (pthread_create(&g_dspark_pool.threads[i],
                           NULL,
                           dspark_parallel_pool_worker_main,
                           (void *)(uintptr_t)i) != 0) {
            g_dspark_pool.stop = true;
            pthread_cond_broadcast(&g_dspark_pool.start_cond);
            for (uint32_t j = 0; j < i; j++) {
                pthread_join(g_dspark_pool.threads[j], NULL);
            }
            pthread_cond_destroy(&g_dspark_pool.done_cond);
            pthread_cond_destroy(&g_dspark_pool.start_cond);
            pthread_mutex_destroy(&g_dspark_pool.mutex);
            memset(&g_dspark_pool, 0, sizeof(g_dspark_pool));
            return false;
        }
    }
    g_dspark_pool.initialized = true;
    return true;
}

static bool dspark_parallel_pool_run(uint32_t n_tasks,
                                     dspark_parallel_worker_fn fn,
                                     void *tasks) {
    if (n_tasks <= 1) {
        if (fn && n_tasks == 1) fn(tasks, 0);
        return true;
    }
    if (getenv("DS4_DSPARK_DISABLE_THREAD_POOL") != NULL) return false;
    if (!dspark_parallel_pool_ensure(n_tasks)) return false;

    pthread_mutex_lock(&g_dspark_pool.mutex);
    g_dspark_pool.fn = fn;
    g_dspark_pool.tasks = tasks;
    g_dspark_pool.n_tasks = n_tasks;
    g_dspark_pool.done_count = 0;
    g_dspark_pool.generation++;
    pthread_cond_broadcast(&g_dspark_pool.start_cond);
    while (g_dspark_pool.done_count < g_dspark_pool.n_threads) {
        pthread_cond_wait(&g_dspark_pool.done_cond, &g_dspark_pool.mutex);
    }
    g_dspark_pool.fn = NULL;
    g_dspark_pool.tasks = NULL;
    g_dspark_pool.n_tasks = 0;
    pthread_mutex_unlock(&g_dspark_pool.mutex);
    return true;
}

static bool dspark_f32_accum_enabled(void) {
    return getenv("DS4_DSPARK_F32_ACCUM") != NULL;
}

typedef struct {
    const unsigned char *weight_data;
    const unsigned char *scale_data;
    const float *in;
    float *out;
    uint64_t row_begin;
    uint64_t row_end;
    uint64_t in_dim;
    uint64_t scale_cols;
    bool f32_accum;
} dspark_linear_f8_task;

static void dspark_linear_f8_rows(const dspark_linear_f8_task *task) {
    if (task->f32_accum) {
        for (uint64_t row = task->row_begin; row < task->row_end; row++) {
            float acc = 0.0f;
            const uint64_t weight_row = row * task->in_dim;
            const uint64_t scale_row = (row / 128u) * task->scale_cols;
            for (uint64_t col = 0; col < task->in_dim; col++) {
                const float wv =
                    dspark_fp8_e4m3_table[task->weight_data[weight_row + col]] *
                    dspark_fp8_e8m0_table[task->scale_data[scale_row + col / 128u]];
                acc += task->in[col] * wv;
            }
            task->out[row] = acc;
        }
        return;
    }
    for (uint64_t row = task->row_begin; row < task->row_end; row++) {
        double acc = 0.0;
        const uint64_t weight_row = row * task->in_dim;
        const uint64_t scale_row = (row / 128u) * task->scale_cols;
        for (uint64_t col = 0; col < task->in_dim; col++) {
            const float wv =
                dspark_fp8_e4m3_table[task->weight_data[weight_row + col]] *
                dspark_fp8_e8m0_table[task->scale_data[scale_row + col / 128u]];
            acc += (double)task->in[col] * (double)wv;
        }
        task->out[row] = (float)acc;
    }
}

static void *dspark_linear_f8_worker(void *arg) {
    dspark_linear_f8_rows((const dspark_linear_f8_task *)arg);
    return NULL;
}

static void dspark_linear_f8_pool_worker(void *tasks, uint32_t index) {
    dspark_linear_f8_rows(&((const dspark_linear_f8_task *)tasks)[index]);
}

typedef struct {
    const unsigned char *weight_data;
    const unsigned char *scale_data;
    const float *in;
    float *out;
    uint64_t row_begin;
    uint64_t row_end;
    uint64_t in_per_group;
    uint64_t out_per_group;
    uint64_t scale_cols;
    bool f32_accum;
} dspark_grouped_linear_f8_task;

static void dspark_grouped_linear_f8_rows(const dspark_grouped_linear_f8_task *task) {
    if (task->f32_accum) {
        for (uint64_t global_row = task->row_begin; global_row < task->row_end; global_row++) {
            const uint64_t group = global_row / task->out_per_group;
            const float *group_in = task->in + group * task->in_per_group;
            const uint64_t weight_row = global_row * task->in_per_group;
            const uint64_t scale_row = (global_row / 128u) * task->scale_cols;
            float acc = 0.0f;
            for (uint64_t col = 0; col < task->in_per_group; col++) {
                const float wv =
                    dspark_fp8_e4m3_table[task->weight_data[weight_row + col]] *
                    dspark_fp8_e8m0_table[task->scale_data[scale_row + col / 128u]];
                acc += group_in[col] * wv;
            }
            task->out[global_row] = acc;
        }
        return;
    }
    for (uint64_t global_row = task->row_begin; global_row < task->row_end; global_row++) {
        const uint64_t group = global_row / task->out_per_group;
        const float *group_in = task->in + group * task->in_per_group;
        const uint64_t weight_row = global_row * task->in_per_group;
        const uint64_t scale_row = (global_row / 128u) * task->scale_cols;
        double acc = 0.0;
        for (uint64_t col = 0; col < task->in_per_group; col++) {
            const float wv =
                dspark_fp8_e4m3_table[task->weight_data[weight_row + col]] *
                dspark_fp8_e8m0_table[task->scale_data[scale_row + col / 128u]];
            acc += (double)group_in[col] * (double)wv;
        }
        task->out[global_row] = (float)acc;
    }
}

static void *dspark_grouped_linear_f8_worker(void *arg) {
    dspark_grouped_linear_f8_rows((const dspark_grouped_linear_f8_task *)arg);
    return NULL;
}

static void dspark_grouped_linear_f8_pool_worker(void *tasks, uint32_t index) {
    dspark_grouped_linear_f8_rows(&((const dspark_grouped_linear_f8_task *)tasks)[index]);
}

typedef struct {
    const unsigned char *weight_data;
    const float *in;
    float *out;
    uint64_t row_begin;
    uint64_t row_end;
    uint64_t in_dim;
    bool f32_accum;
} dspark_linear_bf16_task;

static void dspark_linear_bf16_rows(const dspark_linear_bf16_task *task) {
    if (task->f32_accum) {
        for (uint64_t row = task->row_begin; row < task->row_end; row++) {
            float acc = 0.0f;
            const uint64_t weight_row = row * task->in_dim;
            for (uint64_t col = 0; col < task->in_dim; col++) {
                const uint64_t elem = weight_row + col;
                const uint16_t raw = (uint16_t)task->weight_data[elem * 2u] |
                                     ((uint16_t)task->weight_data[elem * 2u + 1u] << 8);
                acc += task->in[col] * dspark_bf16_table[raw];
            }
            task->out[row] = acc;
        }
        return;
    }
    for (uint64_t row = task->row_begin; row < task->row_end; row++) {
        double acc = 0.0;
        const uint64_t weight_row = row * task->in_dim;
        for (uint64_t col = 0; col < task->in_dim; col++) {
            const uint64_t elem = weight_row + col;
            const uint16_t raw = (uint16_t)task->weight_data[elem * 2u] |
                                 ((uint16_t)task->weight_data[elem * 2u + 1u] << 8);
            acc += (double)task->in[col] * (double)dspark_bf16_table[raw];
        }
        task->out[row] = (float)acc;
    }
}

static void *dspark_linear_bf16_worker(void *arg) {
    dspark_linear_bf16_rows((const dspark_linear_bf16_task *)arg);
    return NULL;
}

static void dspark_linear_bf16_pool_worker(void *tasks, uint32_t index) {
    dspark_linear_bf16_rows(&((const dspark_linear_bf16_task *)tasks)[index]);
}

typedef struct {
    const unsigned char *weight_data;
    const unsigned char *scale_data;
    const float *in;
    float *out;
    uint64_t row_begin;
    uint64_t row_end;
    uint64_t packed_in_dim;
    uint64_t in_dim;
    uint64_t scale_cols;
    bool f32_accum;
} dspark_linear_fp4_task;

static void dspark_linear_fp4_rows(const dspark_linear_fp4_task *task) {
    if (task->f32_accum) {
        for (uint64_t row = task->row_begin; row < task->row_end; row++) {
            float acc = 0.0f;
            const uint64_t packed_row = row * task->packed_in_dim;
            const uint64_t scale_row = row * task->scale_cols;
            for (uint64_t col = 0; col < task->in_dim; col++) {
                const uint8_t packed = task->weight_data[packed_row + col / 2u];
                const uint8_t nibble = (col & 1u) ? (uint8_t)(packed >> 4u) : (uint8_t)(packed & 0x0fu);
                const float wv =
                    dspark_fp4_e2m1_table[nibble] *
                    dspark_fp8_e8m0_table[task->scale_data[scale_row + col / 32u]];
                acc += task->in[col] * wv;
            }
            task->out[row] = acc;
        }
        return;
    }
    for (uint64_t row = task->row_begin; row < task->row_end; row++) {
        double acc = 0.0;
        const uint64_t packed_row = row * task->packed_in_dim;
        const uint64_t scale_row = row * task->scale_cols;
        for (uint64_t col = 0; col < task->in_dim; col++) {
            const uint8_t packed = task->weight_data[packed_row + col / 2u];
            const uint8_t nibble = (col & 1u) ? (uint8_t)(packed >> 4u) : (uint8_t)(packed & 0x0fu);
            const float wv =
                dspark_fp4_e2m1_table[nibble] *
                dspark_fp8_e8m0_table[task->scale_data[scale_row + col / 32u]];
            acc += (double)task->in[col] * (double)wv;
        }
        task->out[row] = (float)acc;
    }
}

static void *dspark_linear_fp4_worker(void *arg) {
    dspark_linear_fp4_rows((const dspark_linear_fp4_task *)arg);
    return NULL;
}

static void dspark_linear_fp4_pool_worker(void *tasks, uint32_t index) {
    dspark_linear_fp4_rows(&((const dspark_linear_fp4_task *)tasks)[index]);
}

float ds4_dflash_fp8_e4m3_to_f32(uint8_t raw) {
    dspark_decode_tables_ensure();
    return dspark_fp8_e4m3_table[raw];
}

float ds4_dflash_fp8_e8m0_to_f32(uint8_t raw) {
    dspark_decode_tables_ensure();
    return dspark_fp8_e8m0_table[raw];
}

float ds4_dflash_fp4_e2m1_to_f32(uint8_t raw) {
    dspark_decode_tables_ensure();
    return dspark_fp4_e2m1_table[raw & 0x0fu];
}

static float read_le_f32(const unsigned char *src) {
    uint32_t bits = ((uint32_t)src[0]) |
                    ((uint32_t)src[1] << 8) |
                    ((uint32_t)src[2] << 16) |
                    ((uint32_t)src[3] << 24);
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

int ds4_dspark_tensor_read_bf16_f32(const ds4_dspark_weights *w,
                                    const ds4_dflash_tensor *tensor,
                                    uint64_t elem_offset,
                                    float *out,
                                    uint64_t n,
                                    char *err,
                                    size_t errlen) {
    const ds4_dspark_shard *shard = NULL;
    const unsigned char *src = NULL;
    uint64_t offset_bytes = 0;
    uint64_t read_bytes = 0;

    if (!w || !w->loaded || !tensor || !out) {
        return dflash_err(err, errlen, "DSpark BF16 read has invalid input");
    }
    if (tensor->shard_index >= w->n_shards) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' references invalid shard %u",
                          tensor->name,
                          tensor->shard_index);
    }
    shard = &w->shards[tensor->shard_index];
    if (!shard->loaded || !shard->map) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' shard is not mapped",
                          tensor->name);
    }
    if (tensor->dtype != DS4_DFLASH_TENSOR_BF16) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' is %s, not BF16",
                          tensor->name, dtype_name(tensor->dtype));
    }
    if (elem_offset > UINT64_MAX / 2u || n > UINT64_MAX / 2u) {
        return dflash_err(err, errlen,
                          "DSpark BF16 read for tensor '%s' is out of bounds",
                          tensor->name);
    }
    offset_bytes = elem_offset * 2u;
    read_bytes = n * 2u;
    if (offset_bytes > tensor->nbytes ||
        read_bytes > tensor->nbytes - offset_bytes ||
        tensor->abs_offset > shard->file_size ||
        offset_bytes > shard->file_size - tensor->abs_offset ||
        read_bytes > shard->file_size - tensor->abs_offset - offset_bytes) {
        return dflash_err(err, errlen,
                          "DSpark BF16 read for tensor '%s' is out of bounds",
                          tensor->name);
    }

    src = (const unsigned char *)shard->map +
          tensor->abs_offset +
          offset_bytes;
    dspark_decode_tables_ensure();
    for (uint64_t i = 0; i < n; i++) {
        const uint16_t raw = (uint16_t)src[i * 2u] |
                             ((uint16_t)src[i * 2u + 1u] << 8);
        out[i] = dspark_bf16_table[raw];
    }
    return 0;
}

int ds4_dspark_weights_read_bf16_f32(const ds4_dspark_weights *w,
                                     const char *name,
                                     uint64_t elem_offset,
                                     float *out,
                                     uint64_t n,
                                     char *err,
                                     size_t errlen) {
    const ds4_dflash_tensor *tensor = ds4_dspark_weights_find_tensor(w, name);
    if (!tensor) {
        return dflash_err(err, errlen,
                          "DSpark safetensors is missing bound tensor '%s'",
                          name ? name : "(null)");
    }
    return ds4_dspark_tensor_read_bf16_f32(w, tensor, elem_offset, out, n, err, errlen);
}

int ds4_dspark_tensor_read_f32(const ds4_dspark_weights *w,
                               const ds4_dflash_tensor *tensor,
                               uint64_t elem_offset,
                               float *out,
                               uint64_t n,
                               char *err,
                               size_t errlen) {
    const ds4_dspark_shard *shard = NULL;
    const unsigned char *src = NULL;
    uint64_t offset_bytes = 0;
    uint64_t read_bytes = 0;

    if (!w || !w->loaded || !tensor || !out) {
        return dflash_err(err, errlen, "DSpark F32 read has invalid input");
    }
    if (tensor->shard_index >= w->n_shards) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' references invalid shard %u",
                          tensor->name,
                          tensor->shard_index);
    }
    shard = &w->shards[tensor->shard_index];
    if (!shard->loaded || !shard->map) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' shard is not mapped",
                          tensor->name);
    }
    if (tensor->dtype != DS4_DFLASH_TENSOR_F32) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' is %s, not F32",
                          tensor->name, dtype_name(tensor->dtype));
    }
    if (elem_offset > UINT64_MAX / 4u || n > UINT64_MAX / 4u) {
        return dflash_err(err, errlen,
                          "DSpark F32 read for tensor '%s' is out of bounds",
                          tensor->name);
    }
    offset_bytes = elem_offset * 4u;
    read_bytes = n * 4u;
    if (offset_bytes > tensor->nbytes ||
        read_bytes > tensor->nbytes - offset_bytes ||
        tensor->abs_offset > shard->file_size ||
        offset_bytes > shard->file_size - tensor->abs_offset ||
        read_bytes > shard->file_size - tensor->abs_offset - offset_bytes) {
        return dflash_err(err, errlen,
                          "DSpark F32 read for tensor '%s' is out of bounds",
                          tensor->name);
    }

    src = (const unsigned char *)shard->map +
          tensor->abs_offset +
          offset_bytes;
    for (uint64_t i = 0; i < n; i++) {
        out[i] = read_le_f32(src + i * 4u);
    }
    return 0;
}

int ds4_dspark_weights_read_f32(const ds4_dspark_weights *w,
                                const char *name,
                                uint64_t elem_offset,
                                float *out,
                                uint64_t n,
                                char *err,
                                size_t errlen) {
    const ds4_dflash_tensor *tensor = ds4_dspark_weights_find_tensor(w, name);
    if (!tensor) {
        return dflash_err(err, errlen,
                          "DSpark safetensors is missing bound tensor '%s'",
                          name ? name : "(null)");
    }
    return ds4_dspark_tensor_read_f32(w, tensor, elem_offset, out, n, err, errlen);
}

static int dspark_tensor_data_ptr(const ds4_dspark_weights *w,
                                  const ds4_dflash_tensor *tensor,
                                  uint64_t byte_offset,
                                  uint64_t nbytes,
                                  const unsigned char **out,
                                  char *err,
                                  size_t errlen) {
    const ds4_dspark_shard *shard = NULL;

    if (!w || !w->loaded || !tensor || !out) {
        return dflash_err(err, errlen, "DSpark tensor data request has invalid input");
    }
    *out = NULL;
    if (tensor->shard_index >= w->n_shards) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' references invalid shard %u",
                          tensor->name,
                          tensor->shard_index);
    }
    shard = &w->shards[tensor->shard_index];
    if (!shard->loaded || !shard->map) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' shard is not mapped",
                          tensor->name);
    }
    if (byte_offset > tensor->nbytes ||
        nbytes > tensor->nbytes - byte_offset ||
        tensor->abs_offset > shard->file_size ||
        byte_offset > shard->file_size - tensor->abs_offset ||
        nbytes > shard->file_size - tensor->abs_offset - byte_offset) {
        return dflash_err(err, errlen,
                          "DSpark tensor '%s' data request is out of bounds",
                          tensor->name);
    }
    *out = (const unsigned char *)shard->map + tensor->abs_offset + byte_offset;
    return 0;
}

int ds4_dspark_linear_f8_f32(const ds4_dspark_weights *w,
                             const ds4_dflash_tensor *weight,
                             const ds4_dflash_tensor *scale,
                             const float *in,
                             float *out,
                             char *err,
                             size_t errlen) {
    const unsigned char *weight_data = NULL;
    const unsigned char *scale_data = NULL;
    uint64_t out_dim = 0;
    uint64_t in_dim = 0;
    uint64_t scale_rows = 0;
    uint64_t scale_cols = 0;

    if (!w || !w->loaded || !weight || !scale || !in || !out) {
        return dflash_err(err, errlen, "DSpark FP8 linear request has invalid input");
    }
    if (weight->dtype != DS4_DFLASH_TENSOR_F8_E4M3 ||
        weight->ndim != 2 ||
        scale->dtype != DS4_DFLASH_TENSOR_F8_E8M0 ||
        scale->ndim != 2) {
        return dflash_err(err, errlen, "DSpark FP8 linear tensor layout is invalid");
    }

    out_dim = weight->shape[0];
    in_dim = weight->shape[1];
    scale_rows = ceil_div_u64(out_dim, 128u);
    scale_cols = ceil_div_u64(in_dim, 128u);
    if (out_dim == 0 || in_dim == 0 ||
        scale->shape[0] != scale_rows ||
        scale->shape[1] != scale_cols) {
        return dflash_err(err, errlen, "DSpark FP8 linear scale layout does not match weight");
    }
    if (dspark_tensor_data_ptr(w,
                               weight,
                               0,
                               weight->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0 ||
        dspark_tensor_data_ptr(w,
                               scale,
                               0,
                               scale->nbytes,
                               &scale_data,
                               err,
                               errlen) != 0) {
        return 1;
    }
    dspark_decode_tables_ensure();

    {
        const uint32_t threads = dspark_parallel_threads_for(out_dim, in_dim);
        const bool f32_accum = dspark_f32_accum_enabled();
        if (threads <= 1) {
            const dspark_linear_f8_task task = {
                .weight_data = weight_data,
                .scale_data = scale_data,
                .in = in,
                .out = out,
                .row_begin = 0,
                .row_end = out_dim,
                .in_dim = in_dim,
                .scale_cols = scale_cols,
                .f32_accum = f32_accum,
            };
            dspark_linear_f8_rows(&task);
        } else {
            dspark_linear_f8_task tasks[DS4_DSPARK_MAX_PARALLEL_THREADS];
            for (uint32_t t = 0; t < threads; t++) {
                tasks[t].weight_data = weight_data;
                tasks[t].scale_data = scale_data;
                tasks[t].in = in;
                tasks[t].out = out;
                tasks[t].row_begin = (out_dim * t) / threads;
                tasks[t].row_end = (out_dim * (uint64_t)(t + 1u)) / threads;
                tasks[t].in_dim = in_dim;
                tasks[t].scale_cols = scale_cols;
                tasks[t].f32_accum = f32_accum;
            }
            if (!dspark_parallel_pool_run(threads,
                                          dspark_linear_f8_pool_worker,
                                          tasks)) {
                pthread_t tids[DS4_DSPARK_MAX_PARALLEL_THREADS];
                uint32_t launched = 0;
                for (uint32_t t = 0; t < threads; t++) {
                    if (pthread_create(&tids[t], NULL, dspark_linear_f8_worker, &tasks[t]) != 0) {
                        for (uint32_t j = 0; j < launched; j++) pthread_join(tids[j], NULL);
                        return dflash_err(err, errlen, "DSpark FP8 linear thread launch failed");
                    }
                    launched++;
                }
                for (uint32_t t = 0; t < launched; t++) pthread_join(tids[t], NULL);
            }
        }
    }
    return 0;
}

int ds4_dspark_grouped_linear_f8_f32(const ds4_dspark_weights *w,
                                     const ds4_dflash_tensor *weight,
                                     const ds4_dflash_tensor *scale,
                                     const float *in,
                                     uint32_t groups,
                                     float *out,
                                     char *err,
                                     size_t errlen) {
    const unsigned char *weight_data = NULL;
    const unsigned char *scale_data = NULL;
    uint64_t out_dim = 0;
    uint64_t in_per_group = 0;
    uint64_t out_per_group = 0;
    uint64_t scale_cols = 0;

    if (!w || !w->loaded || !weight || !scale || !in || groups == 0 || !out) {
        return dflash_err(err, errlen, "DSpark grouped FP8 linear request has invalid input");
    }
    if (weight->dtype != DS4_DFLASH_TENSOR_F8_E4M3 ||
        weight->ndim != 2 ||
        scale->dtype != DS4_DFLASH_TENSOR_F8_E8M0 ||
        scale->ndim != 2) {
        return dflash_err(err, errlen, "DSpark grouped FP8 linear tensor layout is invalid");
    }

    out_dim = weight->shape[0];
    in_per_group = weight->shape[1];
    if (out_dim == 0 || in_per_group == 0 || out_dim % groups != 0) {
        return dflash_err(err, errlen, "DSpark grouped FP8 linear dimensions are invalid");
    }
    out_per_group = out_dim / groups;
    scale_cols = ceil_div_u64(in_per_group, 128u);
    if (scale->shape[0] != ceil_div_u64(out_dim, 128u) ||
        scale->shape[1] != scale_cols) {
        return dflash_err(err, errlen, "DSpark grouped FP8 linear scale layout does not match weight");
    }
    if (dspark_tensor_data_ptr(w,
                               weight,
                               0,
                               weight->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0 ||
        dspark_tensor_data_ptr(w,
                               scale,
                               0,
                               scale->nbytes,
                               &scale_data,
                               err,
                               errlen) != 0) {
        return 1;
    }
    dspark_decode_tables_ensure();

    {
        const uint32_t threads = dspark_parallel_threads_for(out_dim, in_per_group);
        const bool f32_accum = dspark_f32_accum_enabled();
        if (threads <= 1) {
            const dspark_grouped_linear_f8_task task = {
                .weight_data = weight_data,
                .scale_data = scale_data,
                .in = in,
                .out = out,
                .row_begin = 0,
                .row_end = out_dim,
                .in_per_group = in_per_group,
                .out_per_group = out_per_group,
                .scale_cols = scale_cols,
                .f32_accum = f32_accum,
            };
            dspark_grouped_linear_f8_rows(&task);
        } else {
            dspark_grouped_linear_f8_task tasks[DS4_DSPARK_MAX_PARALLEL_THREADS];
            for (uint32_t t = 0; t < threads; t++) {
                tasks[t].weight_data = weight_data;
                tasks[t].scale_data = scale_data;
                tasks[t].in = in;
                tasks[t].out = out;
                tasks[t].row_begin = (out_dim * t) / threads;
                tasks[t].row_end = (out_dim * (uint64_t)(t + 1u)) / threads;
                tasks[t].in_per_group = in_per_group;
                tasks[t].out_per_group = out_per_group;
                tasks[t].scale_cols = scale_cols;
                tasks[t].f32_accum = f32_accum;
            }
            if (!dspark_parallel_pool_run(threads,
                                          dspark_grouped_linear_f8_pool_worker,
                                          tasks)) {
                pthread_t tids[DS4_DSPARK_MAX_PARALLEL_THREADS];
                uint32_t launched = 0;
                for (uint32_t t = 0; t < threads; t++) {
                    if (pthread_create(&tids[t], NULL, dspark_grouped_linear_f8_worker, &tasks[t]) != 0) {
                        for (uint32_t j = 0; j < launched; j++) pthread_join(tids[j], NULL);
                        return dflash_err(err, errlen, "DSpark grouped FP8 linear thread launch failed");
                    }
                    launched++;
                }
                for (uint32_t t = 0; t < launched; t++) pthread_join(tids[t], NULL);
            }
        }
    }
    return 0;
}

int ds4_dspark_linear_bf16_f32(const ds4_dspark_weights *w,
                               const ds4_dflash_tensor *weight,
                               const float *in,
                               float *out,
                               char *err,
                               size_t errlen) {
    const unsigned char *weight_data = NULL;
    uint64_t out_dim = 0;
    uint64_t in_dim = 0;

    if (!w || !w->loaded || !weight || !in || !out) {
        return dflash_err(err, errlen, "DSpark BF16 linear request has invalid input");
    }
    if (weight->dtype != DS4_DFLASH_TENSOR_BF16 ||
        weight->ndim != 2) {
        return dflash_err(err, errlen, "DSpark BF16 linear tensor layout is invalid");
    }
    out_dim = weight->shape[0];
    in_dim = weight->shape[1];
    if (out_dim == 0 || in_dim == 0) {
        return dflash_err(err, errlen, "DSpark BF16 linear dimensions are invalid");
    }
    if (dspark_tensor_data_ptr(w,
                               weight,
                               0,
                               weight->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0) {
        return 1;
    }
    dspark_decode_tables_ensure();

    {
        const uint32_t threads = dspark_parallel_threads_for(out_dim, in_dim);
        const bool f32_accum = dspark_f32_accum_enabled();
        if (threads <= 1) {
            const dspark_linear_bf16_task task = {
                .weight_data = weight_data,
                .in = in,
                .out = out,
                .row_begin = 0,
                .row_end = out_dim,
                .in_dim = in_dim,
                .f32_accum = f32_accum,
            };
            dspark_linear_bf16_rows(&task);
        } else {
            dspark_linear_bf16_task tasks[DS4_DSPARK_MAX_PARALLEL_THREADS];
            for (uint32_t t = 0; t < threads; t++) {
                tasks[t].weight_data = weight_data;
                tasks[t].in = in;
                tasks[t].out = out;
                tasks[t].row_begin = (out_dim * t) / threads;
                tasks[t].row_end = (out_dim * (uint64_t)(t + 1u)) / threads;
                tasks[t].in_dim = in_dim;
                tasks[t].f32_accum = f32_accum;
            }
            if (!dspark_parallel_pool_run(threads,
                                          dspark_linear_bf16_pool_worker,
                                          tasks)) {
                pthread_t tids[DS4_DSPARK_MAX_PARALLEL_THREADS];
                uint32_t launched = 0;
                for (uint32_t t = 0; t < threads; t++) {
                    if (pthread_create(&tids[t], NULL, dspark_linear_bf16_worker, &tasks[t]) != 0) {
                        for (uint32_t j = 0; j < launched; j++) pthread_join(tids[j], NULL);
                        return dflash_err(err, errlen, "DSpark BF16 linear thread launch failed");
                    }
                    launched++;
                }
                for (uint32_t t = 0; t < launched; t++) pthread_join(tids[t], NULL);
            }
        }
    }
    return 0;
}

int ds4_dspark_linear_fp4_f32(const ds4_dspark_weights *w,
                              const ds4_dflash_tensor *weight,
                              const ds4_dflash_tensor *scale,
                              const float *in,
                              float *out,
                              char *err,
                              size_t errlen) {
    const unsigned char *weight_data = NULL;
    const unsigned char *scale_data = NULL;
    uint64_t out_dim = 0;
    uint64_t packed_in_dim = 0;
    uint64_t in_dim = 0;
    uint64_t scale_cols = 0;

    if (!w || !w->loaded || !weight || !scale || !in || !out) {
        return dflash_err(err, errlen, "DSpark FP4 linear request has invalid input");
    }
    if (weight->dtype != DS4_DFLASH_TENSOR_I8 ||
        weight->ndim != 2 ||
        scale->dtype != DS4_DFLASH_TENSOR_F8_E8M0 ||
        scale->ndim != 2) {
        return dflash_err(err, errlen, "DSpark FP4 linear tensor layout is invalid");
    }

    out_dim = weight->shape[0];
    packed_in_dim = weight->shape[1];
    if (out_dim == 0 || packed_in_dim == 0 || packed_in_dim > UINT64_MAX / 2u) {
        return dflash_err(err, errlen, "DSpark FP4 linear dimensions are invalid");
    }
    in_dim = packed_in_dim * 2u;
    scale_cols = ceil_div_u64(in_dim, 32u);
    if (scale->shape[0] != out_dim ||
        scale->shape[1] != scale_cols) {
        return dflash_err(err, errlen, "DSpark FP4 linear scale layout does not match weight");
    }
    if (dspark_tensor_data_ptr(w,
                               weight,
                               0,
                               weight->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0 ||
        dspark_tensor_data_ptr(w,
                               scale,
                               0,
                               scale->nbytes,
                               &scale_data,
                               err,
                               errlen) != 0) {
        return 1;
    }
    dspark_decode_tables_ensure();

    {
        const uint32_t threads = dspark_parallel_threads_for(out_dim, in_dim);
        const bool f32_accum = dspark_f32_accum_enabled();
        if (threads <= 1) {
            const dspark_linear_fp4_task task = {
                .weight_data = weight_data,
                .scale_data = scale_data,
                .in = in,
                .out = out,
                .row_begin = 0,
                .row_end = out_dim,
                .packed_in_dim = packed_in_dim,
                .in_dim = in_dim,
                .scale_cols = scale_cols,
                .f32_accum = f32_accum,
            };
            dspark_linear_fp4_rows(&task);
        } else {
            dspark_linear_fp4_task tasks[DS4_DSPARK_MAX_PARALLEL_THREADS];
            for (uint32_t t = 0; t < threads; t++) {
                tasks[t].weight_data = weight_data;
                tasks[t].scale_data = scale_data;
                tasks[t].in = in;
                tasks[t].out = out;
                tasks[t].row_begin = (out_dim * t) / threads;
                tasks[t].row_end = (out_dim * (uint64_t)(t + 1u)) / threads;
                tasks[t].packed_in_dim = packed_in_dim;
                tasks[t].in_dim = in_dim;
                tasks[t].scale_cols = scale_cols;
                tasks[t].f32_accum = f32_accum;
            }
            if (!dspark_parallel_pool_run(threads,
                                          dspark_linear_fp4_pool_worker,
                                          tasks)) {
                pthread_t tids[DS4_DSPARK_MAX_PARALLEL_THREADS];
                uint32_t launched = 0;
                for (uint32_t t = 0; t < threads; t++) {
                    if (pthread_create(&tids[t], NULL, dspark_linear_fp4_worker, &tasks[t]) != 0) {
                        for (uint32_t j = 0; j < launched; j++) pthread_join(tids[j], NULL);
                        return dflash_err(err, errlen, "DSpark FP4 linear thread launch failed");
                    }
                    launched++;
                }
                for (uint32_t t = 0; t < launched; t++) pthread_join(tids[t], NULL);
            }
        }
    }
    return 0;
}

int ds4_dspark_rms_norm_bf16(const ds4_dspark_weights *w,
                             const ds4_dflash_tensor *weight,
                             const float *in,
                             uint64_t n,
                             float *out,
                             char *err,
                             size_t errlen) {
    const unsigned char *weight_data = NULL;
    double ss = 0.0;
    float inv_rms = 0.0f;

    if (!w || !w->loaded || !weight || !in || !out || n == 0) {
        return dflash_err(err, errlen, "DSpark BF16 RMSNorm request has invalid input");
    }
    if (weight->dtype != DS4_DFLASH_TENSOR_BF16 ||
        weight->ndim != 1 ||
        weight->shape[0] != n) {
        return dflash_err(err, errlen, "DSpark BF16 RMSNorm tensor layout is invalid");
    }
    if (dspark_tensor_data_ptr(w,
                               weight,
                               0,
                               weight->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0) {
        return 1;
    }
    dspark_decode_tables_ensure();

    for (uint64_t i = 0; i < n; i++) {
        ss += (double)in[i] * (double)in[i];
    }
    inv_rms = 1.0f / sqrtf((float)(ss / (double)n) + DS4_DFLASH_RMS_EPS);
    for (uint64_t i = 0; i < n; i++) {
        const uint16_t raw = (uint16_t)weight_data[i * 2u] |
                             ((uint16_t)weight_data[i * 2u + 1u] << 8);
        out[i] = in[i] * inv_rms * dspark_bf16_table[raw];
    }
    return 0;
}

int ds4_dspark_rms_norm_f32(const float *in,
                            uint64_t n,
                            float *out,
                            char *err,
                            size_t errlen) {
    double ss = 0.0;
    float inv_rms = 0.0f;

    if (!in || n == 0 || !out) {
        return dflash_err(err, errlen, "DSpark F32 RMSNorm request has invalid input");
    }
    for (uint64_t i = 0; i < n; i++) {
        ss += (double)in[i] * (double)in[i];
    }
    inv_rms = 1.0f / sqrtf((float)(ss / (double)n) + DS4_DFLASH_RMS_EPS);
    for (uint64_t i = 0; i < n; i++) {
        out[i] = in[i] * inv_rms;
    }
    return 0;
}

static float dspark_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

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
                               size_t errlen) {
    const unsigned char *fn_data = NULL;
    const unsigned char *scale_data = NULL;
    const unsigned char *base_data = NULL;
    uint64_t hc_dim = 0;
    uint64_t mix_hc = 0;
    double ss = 0.0;
    float inv_rms = 0.0f;
    float pre_scale = 0.0f;
    float post_scale = 0.0f;
    float comb_scale = 0.0f;

    if (!w || !w->loaded || !fn || !scale || !base || !residual_hc ||
        hidden == 0 || hc_mult == 0 || !collapsed || !split || eps <= 0.0f) {
        return dflash_err(err, errlen, "DSpark HC collapse request has invalid input");
    }
    if (hc_mult > UINT64_MAX / hidden ||
        (uint64_t)(hc_mult + 2u) > UINT64_MAX / hc_mult) {
        return dflash_err(err, errlen, "DSpark HC collapse dimensions overflow");
    }
    hc_dim = (uint64_t)hc_mult * hidden;
    mix_hc = (uint64_t)(hc_mult + 2u) * hc_mult;
    if (fn->dtype != DS4_DFLASH_TENSOR_F32 ||
        fn->ndim != 2 ||
        fn->shape[0] != mix_hc ||
        fn->shape[1] != hc_dim ||
        scale->dtype != DS4_DFLASH_TENSOR_F32 ||
        scale->ndim != 1 ||
        scale->shape[0] != 3 ||
        base->dtype != DS4_DFLASH_TENSOR_F32 ||
        base->ndim != 1 ||
        base->shape[0] != mix_hc) {
        return dflash_err(err, errlen, "DSpark HC tensor layout is invalid");
    }
    if (dspark_tensor_data_ptr(w, fn, 0, fn->nbytes, &fn_data, err, errlen) != 0 ||
        dspark_tensor_data_ptr(w, scale, 0, scale->nbytes, &scale_data, err, errlen) != 0 ||
        dspark_tensor_data_ptr(w, base, 0, base->nbytes, &base_data, err, errlen) != 0) {
        return 1;
    }

    for (uint64_t i = 0; i < hc_dim; i++) {
        ss += (double)residual_hc[i] * (double)residual_hc[i];
    }
    inv_rms = 1.0f / sqrtf((float)(ss / (double)hc_dim) + DS4_DFLASH_RMS_EPS);

    for (uint64_t row = 0; row < mix_hc; row++) {
        double acc = 0.0;
        const uint64_t fn_row = row * hc_dim;
        for (uint64_t col = 0; col < hc_dim; col++) {
            const float wv = read_le_f32(fn_data + (fn_row + col) * 4u);
            acc += (double)(residual_hc[col] * inv_rms) * (double)wv;
        }
        split[row] = (float)acc;
    }

    pre_scale = read_le_f32(scale_data);
    post_scale = read_le_f32(scale_data + 4u);
    comb_scale = read_le_f32(scale_data + 8u);
    for (uint32_t h = 0; h < hc_mult; h++) {
        const float pre_b = read_le_f32(base_data + (uint64_t)h * 4u);
        const float post_b = read_le_f32(base_data + (uint64_t)(hc_mult + h) * 4u);
        split[h] = dspark_sigmoidf(split[h] * pre_scale + pre_b) + eps;
        split[hc_mult + h] =
            2.0f * dspark_sigmoidf(split[hc_mult + h] * post_scale + post_b);
    }

    for (uint32_t row = 0; row < hc_mult; row++) {
        float max_v = -FLT_MAX;
        float sum = 0.0f;
        for (uint32_t col = 0; col < hc_mult; col++) {
            const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
            const float b = read_le_f32(base_data + idx * 4u);
            const float v = split[idx] * comb_scale + b;
            split[idx] = v;
            if (v > max_v) max_v = v;
        }
        for (uint32_t col = 0; col < hc_mult; col++) {
            const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
            const float v = expf(split[idx] - max_v);
            split[idx] = v;
            sum += v;
        }
        if (sum == 0.0f) return dflash_err(err, errlen, "DSpark HC softmax underflowed");
        for (uint32_t col = 0; col < hc_mult; col++) {
            const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
            split[idx] = split[idx] / sum + eps;
        }
    }

    for (uint32_t col = 0; col < hc_mult; col++) {
        float sum = eps;
        for (uint32_t row = 0; row < hc_mult; row++) {
            const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
            sum += split[idx];
        }
        for (uint32_t row = 0; row < hc_mult; row++) {
            const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
            split[idx] /= sum;
        }
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (uint32_t row = 0; row < hc_mult; row++) {
            float sum = eps;
            for (uint32_t col = 0; col < hc_mult; col++) {
                const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
                sum += split[idx];
            }
            for (uint32_t col = 0; col < hc_mult; col++) {
                const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
                split[idx] /= sum;
            }
        }
        for (uint32_t col = 0; col < hc_mult; col++) {
            float sum = eps;
            for (uint32_t row = 0; row < hc_mult; row++) {
                const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
                sum += split[idx];
            }
            for (uint32_t row = 0; row < hc_mult; row++) {
                const uint64_t idx = (uint64_t)2u * hc_mult + (uint64_t)row * hc_mult + col;
                split[idx] /= sum;
            }
        }
    }

    for (uint32_t d = 0; d < hidden; d++) {
        double acc = 0.0;
        for (uint32_t h = 0; h < hc_mult; h++) {
            acc += (double)residual_hc[(uint64_t)h * hidden + d] * (double)split[h];
        }
        collapsed[d] = (float)acc;
    }
    return 0;
}

int ds4_dspark_hc_expand_f32(const float *block_out,
                             const float *residual_hc,
                             const float *split,
                             uint32_t hidden,
                             uint32_t hc_mult,
                             float *out_hc,
                             char *err,
                             size_t errlen) {
    if (!block_out || !residual_hc || !split || hidden == 0 || hc_mult == 0 || !out_hc) {
        return dflash_err(err, errlen, "DSpark HC expand request has invalid input");
    }
    if (hc_mult > UINT64_MAX / hidden) {
        return dflash_err(err, errlen, "DSpark HC expand dimensions overflow");
    }

    for (uint32_t dst_hc = 0; dst_hc < hc_mult; dst_hc++) {
        for (uint32_t d = 0; d < hidden; d++) {
            double acc = (double)block_out[d] * (double)split[hc_mult + dst_hc];
            for (uint32_t src_hc = 0; src_hc < hc_mult; src_hc++) {
                const uint64_t comb_idx = (uint64_t)2u * hc_mult +
                                          (uint64_t)src_hc * hc_mult +
                                          dst_hc;
                acc += (double)split[comb_idx] *
                       (double)residual_hc[(uint64_t)src_hc * hidden + d];
            }
            out_hc[(uint64_t)dst_hc * hidden + d] = (float)acc;
        }
    }
    return 0;
}

static float dspark_sqrt_softplus(float x) {
    float sp = 0.0f;
    if (x > 0.0f) {
        sp = x + log1pf(expf(-x));
    } else {
        sp = log1pf(expf(x));
    }
    return sqrtf(sp);
}

int ds4_dspark_moe_gate_topk_f32(const ds4_dspark_weights *w,
                                 const ds4_dflash_tensor *weight,
                                 const ds4_dflash_tensor *bias,
                                 const float *in,
                                 uint32_t topk,
                                 float route_scale,
                                 uint32_t *indices,
                                 float *weights,
                                 char *err,
                                 size_t errlen) {
    const unsigned char *weight_data = NULL;
    const unsigned char *bias_data = NULL;
    uint64_t n_experts = 0;
    uint64_t hidden = 0;
    float *scores = NULL;
    float *adjusted = NULL;
    bool *chosen = NULL;
    float *scores_heap = NULL;
    float *adjusted_heap = NULL;
    bool *chosen_heap = NULL;
    float scores_stack[DS4_DSPARK_STACK_ROUTED_EXPERTS];
    float adjusted_stack[DS4_DSPARK_STACK_ROUTED_EXPERTS];
    bool chosen_stack[DS4_DSPARK_STACK_ROUTED_EXPERTS];
    double selected_sum = 0.0;
    int rc = 1;

    if (!w || !w->loaded || !weight || !bias || !in || topk == 0 ||
        route_scale <= 0.0f || !indices || !weights) {
        return dflash_err(err, errlen, "DSpark MoE gate request has invalid input");
    }
    if (weight->dtype != DS4_DFLASH_TENSOR_BF16 ||
        weight->ndim != 2 ||
        bias->dtype != DS4_DFLASH_TENSOR_F32 ||
        bias->ndim != 1 ||
        bias->shape[0] != weight->shape[0] ||
        weight->shape[0] > UINT32_MAX) {
        return dflash_err(err, errlen, "DSpark MoE gate tensor layout is invalid");
    }
    n_experts = weight->shape[0];
    hidden = weight->shape[1];
    if (n_experts == 0 || hidden == 0 || topk > n_experts) {
        return dflash_err(err, errlen, "DSpark MoE gate dimensions are invalid");
    }
    if (dspark_tensor_data_ptr(w,
                               weight,
                               0,
                               weight->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0 ||
        dspark_tensor_data_ptr(w,
                               bias,
                               0,
                               bias->nbytes,
                               &bias_data,
                               err,
                               errlen) != 0) {
        return 1;
    }

    if (n_experts <= DS4_DSPARK_STACK_ROUTED_EXPERTS) {
        scores = scores_stack;
        adjusted = adjusted_stack;
        chosen = chosen_stack;
        memset(chosen, 0, (size_t)n_experts * sizeof(chosen[0]));
    } else {
        scores_heap = malloc((size_t)n_experts * sizeof(scores_heap[0]));
        adjusted_heap = malloc((size_t)n_experts * sizeof(adjusted_heap[0]));
        chosen_heap = calloc((size_t)n_experts, sizeof(chosen_heap[0]));
        scores = scores_heap;
        adjusted = adjusted_heap;
        chosen = chosen_heap;
        if (!scores || !adjusted || !chosen) {
            dflash_err(err, errlen, "out of memory computing DSpark MoE gate");
            goto out;
        }
    }
    dspark_decode_tables_ensure();

    for (uint64_t expert = 0; expert < n_experts; expert++) {
        double logit = 0.0;
        const uint64_t row = expert * hidden;
        for (uint64_t col = 0; col < hidden; col++) {
            const uint64_t elem = row + col;
            const uint16_t raw = (uint16_t)weight_data[elem * 2u] |
                                 ((uint16_t)weight_data[elem * 2u + 1u] << 8);
            logit += (double)in[col] * (double)dspark_bf16_table[raw];
        }
        scores[expert] = dspark_sqrt_softplus((float)logit);
        adjusted[expert] = scores[expert] + read_le_f32(bias_data + expert * 4u);
    }

    for (uint32_t k = 0; k < topk; k++) {
        uint64_t best = UINT64_MAX;
        float best_score = -FLT_MAX;
        for (uint64_t expert = 0; expert < n_experts; expert++) {
            if (chosen[expert]) continue;
            if (best == UINT64_MAX || adjusted[expert] > best_score) {
                best = expert;
                best_score = adjusted[expert];
            }
        }
        if (best == UINT64_MAX) {
            dflash_err(err, errlen, "DSpark MoE gate could not select top-k expert");
            goto out;
        }
        chosen[best] = true;
        indices[k] = (uint32_t)best;
        weights[k] = scores[best];
        selected_sum += (double)weights[k];
    }
    if (selected_sum <= 0.0) {
        dflash_err(err, errlen, "DSpark MoE gate selected zero total score");
        goto out;
    }
    for (uint32_t k = 0; k < topk; k++) {
        weights[k] = (float)((double)weights[k] / selected_sum) * route_scale;
    }
    rc = 0;

out:
    free(scores_heap);
    free(adjusted_heap);
    free(chosen_heap);
    return rc;
}

int ds4_dspark_swiglu_f32(const float *gate,
                          const float *up,
                          uint64_t n,
                          float limit,
                          float *out,
                          char *err,
                          size_t errlen) {
    if (!gate || !up || n == 0 || !out) {
        return dflash_err(err, errlen, "DSpark SwiGLU request has invalid input");
    }
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (limit > 0.0f) {
            if (g > limit) g = limit;
            if (u > limit) u = limit;
            if (u < -limit) u = -limit;
        }
        out[i] = (g / (1.0f + expf(-g))) * u;
    }
    return 0;
}

static float dspark_rope_correction_dim(float rotations,
                                        uint32_t dim,
                                        float base,
                                        uint32_t max_seq_len) {
    const float two_pi = 6.2831853071795864769f;
    return (float)dim * logf((float)max_seq_len / (rotations * two_pi)) /
           (2.0f * logf(base));
}

static float dspark_rope_ramp(float x, float lo, float hi) {
    float v = 0.0f;
    if (lo == hi) hi += 0.001f;
    v = (x - lo) / (hi - lo);
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static float dspark_rope_inv_freq(uint32_t pair,
                                  uint32_t rope_dim,
                                  float theta,
                                  uint32_t original_seq_len,
                                  float factor,
                                  float beta_fast,
                                  float beta_slow) {
    float freq = 1.0f / powf(theta, (float)(pair * 2u) / (float)rope_dim);

    if (original_seq_len != 0 && factor > 1.0f && theta > 1.0f) {
        float lo = floorf(dspark_rope_correction_dim(beta_fast,
                                                     rope_dim,
                                                     theta,
                                                     original_seq_len));
        float hi = ceilf(dspark_rope_correction_dim(beta_slow,
                                                    rope_dim,
                                                    theta,
                                                    original_seq_len));
        const float max_dim = rope_dim > 0 ? (float)(rope_dim - 1u) : 0.0f;
        float ramp = 0.0f;
        if (lo < 0.0f) lo = 0.0f;
        if (hi < 0.0f) hi = 0.0f;
        if (lo > max_dim) lo = max_dim;
        if (hi > max_dim) hi = max_dim;
        ramp = dspark_rope_ramp((float)pair, lo, hi);
        freq = freq * (1.0f - ramp) + (freq / factor) * ramp;
    }
    return freq;
}

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
                                      size_t errlen) {
    if (!x || n_heads == 0 || head_dim == 0 || rope_dim == 0 ||
        rope_dim > head_dim || (rope_dim % 2u) != 0 || theta <= 1.0f) {
        return dflash_err(err, errlen, "DSpark partial RoPE request has invalid input");
    }
    if (factor <= 0.0f) factor = 1.0f;
    if (beta_fast <= 0.0f) beta_fast = 32.0f;
    if (beta_slow <= 0.0f) beta_slow = 1.0f;

    for (uint32_t head = 0; head < n_heads; head++) {
        float *rope = x + (uint64_t)head * head_dim + (head_dim - rope_dim);
        for (uint32_t i = 0; i < rope_dim; i += 2u) {
            const uint32_t pair = i / 2u;
            const float freq = dspark_rope_inv_freq(pair,
                                                    rope_dim,
                                                    theta,
                                                    original_seq_len,
                                                    factor,
                                                    beta_fast,
                                                    beta_slow);
            const float angle = (float)((double)position * (double)freq);
            const float c = cosf(angle);
            float s = sinf(angle);
            const float x0 = rope[i];
            const float x1 = rope[i + 1u];
            if (inverse) s = -s;
            rope[i] = x0 * c - x1 * s;
            rope[i + 1u] = x1 * c + x0 * s;
        }
    }
    return 0;
}

int ds4_dspark_sparse_attention_one_f32(const ds4_dspark_weights *w,
                                        const ds4_dflash_tensor *sink,
                                        const float *q,
                                        const float *kv,
                                        uint32_t n_heads,
                                        uint32_t head_dim,
                                        float *out,
                                        char *err,
                                        size_t errlen) {
    const unsigned char *sink_data = NULL;
    const float scale = head_dim ? 1.0f / sqrtf((float)head_dim) : 0.0f;

    if (!w || !w->loaded || !sink || !q || !kv || n_heads == 0 ||
        head_dim == 0 || !out) {
        return dflash_err(err, errlen, "DSpark sparse attention request has invalid input");
    }
    if (sink->dtype != DS4_DFLASH_TENSOR_F32 ||
        sink->ndim != 1 ||
        sink->shape[0] != n_heads) {
        return dflash_err(err, errlen, "DSpark sparse attention sink tensor layout is invalid");
    }
    if (dspark_tensor_data_ptr(w,
                               sink,
                               0,
                               sink->nbytes,
                               &sink_data,
                               err,
                               errlen) != 0) {
        return 1;
    }

    for (uint32_t head = 0; head < n_heads; head++) {
        const float *qh = q + (uint64_t)head * head_dim;
        float *oh = out + (uint64_t)head * head_dim;
        double dot = 0.0;
        float score = 0.0f;
        float sink_score = 0.0f;
        float max_score = 0.0f;
        float row_weight = 0.0f;
        float sink_weight = 0.0f;
        float denom = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += (double)qh[d] * (double)kv[d];
        }
        score = (float)dot * scale;
        sink_score = read_le_f32(sink_data + (uint64_t)head * 4u);
        max_score = score > sink_score ? score : sink_score;
        row_weight = expf(score - max_score);
        sink_weight = expf(sink_score - max_score);
        denom = row_weight + sink_weight;
        row_weight = denom == 0.0f ? 0.0f : row_weight / denom;
        for (uint32_t d = 0; d < head_dim; d++) {
            oh[d] = kv[d] * row_weight;
        }
    }
    return 0;
}

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
                                          size_t errlen) {
    const unsigned char *sink_data = NULL;
    const float scale = head_dim ? 1.0f / sqrtf((float)head_dim) : 0.0f;

    if (!w || !w->loaded || !sink || !q || !kv || q_len == 0 || kv_len == 0 ||
        n_heads == 0 || head_dim == 0 || !out) {
        return dflash_err(err, errlen, "DSpark block attention request has invalid input");
    }
    if (sink->dtype != DS4_DFLASH_TENSOR_F32 ||
        sink->ndim != 1 ||
        sink->shape[0] != n_heads) {
        return dflash_err(err, errlen, "DSpark block attention sink tensor layout is invalid");
    }
    if (dspark_tensor_data_ptr(w,
                               sink,
                               0,
                               sink->nbytes,
                               &sink_data,
                               err,
                               errlen) != 0) {
        return 1;
    }

    for (uint32_t row = 0; row < q_len; row++) {
        for (uint32_t head = 0; head < n_heads; head++) {
            const float *qh = q + ((uint64_t)row * n_heads + head) * head_dim;
            float *oh = out + ((uint64_t)row * n_heads + head) * head_dim;
            const float sink_score = read_le_f32(sink_data + (uint64_t)head * 4u);
            float max_score = sink_score;
            double denom = 0.0;

            for (uint32_t kv_row = 0; kv_row < kv_len; kv_row++) {
                const float *kh = kv + (uint64_t)kv_row * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)qh[d] * (double)kh[d];
                }
                const float score = (float)dot * scale;
                if (score > max_score) max_score = score;
            }

            for (uint32_t d = 0; d < head_dim; d++) oh[d] = 0.0f;
            denom = exp((double)sink_score - (double)max_score);
            for (uint32_t kv_row = 0; kv_row < kv_len; kv_row++) {
                const float *kh = kv + (uint64_t)kv_row * head_dim;
                double dot = 0.0;
                double weight = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += (double)qh[d] * (double)kh[d];
                }
                weight = exp(dot * (double)scale - (double)max_score);
                denom += weight;
                for (uint32_t d = 0; d < head_dim; d++) {
                    oh[d] += (float)((double)kh[d] * weight);
                }
            }
            if (denom != 0.0) {
                const float inv_denom = (float)(1.0 / denom);
                for (uint32_t d = 0; d < head_dim; d++) oh[d] *= inv_denom;
            }
        }
    }
    return 0;
}

void ds4_dspark_stage_stats_init(ds4_dspark_stage_stats *stats) {
    if (stats) memset(stats, 0, sizeof(*stats));
}

static double dspark_checksum_f32(const float *x, uint64_t n) {
    double sum = 0.0;
    if (!x) return 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += (double)x[i] * (double)((i % 17u) + 1u);
    }
    return sum;
}

static const ds4_dflash_tensor *dspark_find_stage_tensor_required(
    const ds4_dspark_weights *w,
    uint32_t stage,
    const char *suffix,
    char *err,
        size_t errlen) {
    char name[DS4_DFLASH_MAX_TENSOR_NAME];
    const ds4_dflash_tensor *tensor = NULL;

    if (dspark_stage_tensor_name(name, sizeof(name), stage, suffix, err, errlen) != 0) {
        return NULL;
    }
    tensor = ds4_dspark_weights_find_tensor(w, name);
    if (!tensor) {
        dflash_err(err, errlen, "DSpark graph is missing bound tensor '%s'", name);
        return NULL;
    }
    return tensor;
}

typedef struct {
    const ds4_dflash_tensor *hc_attn_fn;
    const ds4_dflash_tensor *hc_attn_scale;
    const ds4_dflash_tensor *hc_attn_base;
    const ds4_dflash_tensor *attn_norm_weight;
    const ds4_dflash_tensor *attn_wq_a_weight;
    const ds4_dflash_tensor *attn_wq_a_scale;
    const ds4_dflash_tensor *attn_q_norm_weight;
    const ds4_dflash_tensor *attn_wq_b_weight;
    const ds4_dflash_tensor *attn_wq_b_scale;
    const ds4_dflash_tensor *attn_wkv_weight;
    const ds4_dflash_tensor *attn_wkv_scale;
    const ds4_dflash_tensor *attn_kv_norm_weight;
    const ds4_dflash_tensor *attn_sink;
    const ds4_dflash_tensor *attn_wo_a_weight;
    const ds4_dflash_tensor *attn_wo_a_scale;
    const ds4_dflash_tensor *attn_wo_b_weight;
    const ds4_dflash_tensor *attn_wo_b_scale;
    const ds4_dflash_tensor *hc_ffn_fn;
    const ds4_dflash_tensor *hc_ffn_scale;
    const ds4_dflash_tensor *hc_ffn_base;
    const ds4_dflash_tensor *ffn_norm_weight;
    const ds4_dflash_tensor *ffn_gate_weight;
    const ds4_dflash_tensor *ffn_gate_bias;
    const ds4_dflash_tensor *shared_w1_weight;
    const ds4_dflash_tensor *shared_w1_scale;
    const ds4_dflash_tensor *shared_w3_weight;
    const ds4_dflash_tensor *shared_w3_scale;
    const ds4_dflash_tensor *shared_w2_weight;
    const ds4_dflash_tensor *shared_w2_scale;
} dspark_stage_block_tensors;

static int dspark_stage_block_tensors_bind(dspark_stage_block_tensors *t,
                                           const ds4_dspark_weights *w,
                                           uint32_t stage,
                                           char *err,
                                           size_t errlen) {
    if (!t) return dflash_err(err, errlen, "invalid DSpark stage tensor cache");
    memset(t, 0, sizeof(*t));
#define DSPARK_BIND_STAGE_TENSOR(field, suffix)                                      \
    do {                                                                            \
        t->field = dspark_find_stage_tensor_required(w, stage, suffix, err, errlen); \
        if (!t->field) return 1;                                                    \
    } while (0)
    DSPARK_BIND_STAGE_TENSOR(hc_attn_fn, "hc_attn_fn");
    DSPARK_BIND_STAGE_TENSOR(hc_attn_scale, "hc_attn_scale");
    DSPARK_BIND_STAGE_TENSOR(hc_attn_base, "hc_attn_base");
    DSPARK_BIND_STAGE_TENSOR(attn_norm_weight, "attn_norm.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wq_a_weight, "attn.wq_a.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wq_a_scale, "attn.wq_a.scale");
    DSPARK_BIND_STAGE_TENSOR(attn_q_norm_weight, "attn.q_norm.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wq_b_weight, "attn.wq_b.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wq_b_scale, "attn.wq_b.scale");
    DSPARK_BIND_STAGE_TENSOR(attn_wkv_weight, "attn.wkv.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wkv_scale, "attn.wkv.scale");
    DSPARK_BIND_STAGE_TENSOR(attn_kv_norm_weight, "attn.kv_norm.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_sink, "attn.attn_sink");
    DSPARK_BIND_STAGE_TENSOR(attn_wo_a_weight, "attn.wo_a.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wo_a_scale, "attn.wo_a.scale");
    DSPARK_BIND_STAGE_TENSOR(attn_wo_b_weight, "attn.wo_b.weight");
    DSPARK_BIND_STAGE_TENSOR(attn_wo_b_scale, "attn.wo_b.scale");
    DSPARK_BIND_STAGE_TENSOR(hc_ffn_fn, "hc_ffn_fn");
    DSPARK_BIND_STAGE_TENSOR(hc_ffn_scale, "hc_ffn_scale");
    DSPARK_BIND_STAGE_TENSOR(hc_ffn_base, "hc_ffn_base");
    DSPARK_BIND_STAGE_TENSOR(ffn_norm_weight, "ffn_norm.weight");
    DSPARK_BIND_STAGE_TENSOR(ffn_gate_weight, "ffn.gate.weight");
    DSPARK_BIND_STAGE_TENSOR(ffn_gate_bias, "ffn.gate.bias");
    DSPARK_BIND_STAGE_TENSOR(shared_w1_weight, "ffn.shared_experts.w1.weight");
    DSPARK_BIND_STAGE_TENSOR(shared_w1_scale, "ffn.shared_experts.w1.scale");
    DSPARK_BIND_STAGE_TENSOR(shared_w3_weight, "ffn.shared_experts.w3.weight");
    DSPARK_BIND_STAGE_TENSOR(shared_w3_scale, "ffn.shared_experts.w3.scale");
    DSPARK_BIND_STAGE_TENSOR(shared_w2_weight, "ffn.shared_experts.w2.weight");
    DSPARK_BIND_STAGE_TENSOR(shared_w2_scale, "ffn.shared_experts.w2.scale");
#undef DSPARK_BIND_STAGE_TENSOR
    return 0;
}

static int dspark_stage_expert_tensor_name(char *out,
                                           size_t outlen,
                                           uint32_t stage,
                                           uint32_t expert,
                                           const char *suffix,
                                           char *err,
                                           size_t errlen) {
    int n = snprintf(out, outlen, "mtp.%u.ffn.experts.%u.%s", stage, expert, suffix);
    if (n < 0 || (size_t)n >= outlen) {
        return dflash_err(err, errlen, "DSpark expert tensor name is too long");
    }
    return 0;
}

int ds4_dspark_run_stage_reference(const ds4_dspark_weights *w,
                                   const ds4_dspark_config *cfg,
                                   uint32_t stage,
                                   uint64_t position,
                                   const float *in_hc,
                                   float *out_hc,
                                   ds4_dspark_stage_stats *stats,
                                   char *err,
                                   size_t errlen) {
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t hc_mult = cfg ? cfg->hc_mult : 0;
    const uint64_t hc_dim = hidden * hc_mult;
    const uint64_t mix_hc = (hc_mult + 2u) * hc_mult;
    const uint64_t q_dim = cfg ? (uint64_t)cfg->num_attention_heads * cfg->head_dim : 0;
    const uint64_t attn_mid_dim = cfg ? (uint64_t)cfg->o_groups * cfg->o_lora_rank : 0;
    float *attn_split = NULL;
    float *attn_x = NULL;
    float *attn_norm = NULL;
    float *q_lora = NULL;
    float *q_lora_norm = NULL;
    float *q = NULL;
    float *wkv = NULL;
    float *wkv_norm = NULL;
    float *attn_heads = NULL;
    float *attn_mid = NULL;
    float *attn_out = NULL;
    float *attn_hc = NULL;
    float *ffn_split = NULL;
    float *ffn_x = NULL;
    float *ffn_norm = NULL;
    uint32_t route_indices[DS4_DSPARK_MAX_ACTIVATED_EXPERTS];
    float route_weights[DS4_DSPARK_MAX_ACTIVATED_EXPERTS];
    float *shared_gate = NULL;
    float *shared_up = NULL;
    float *shared_mid = NULL;
    float *shared_out = NULL;
    float *routed_gate = NULL;
    float *routed_up = NULL;
    float *routed_mid = NULL;
    float *routed_down = NULL;
    float *routed_sum = NULL;
    float *ffn_out = NULL;
    int rc = 1;

    if (stats) ds4_dspark_stage_stats_init(stats);
    if (!w || !w->loaded || !cfg || !cfg->loaded || !in_hc || !out_hc ||
        stage >= cfg->n_mtp_stages ||
        hidden == 0 || hc_mult == 0 || hc_dim == 0 || mix_hc == 0 ||
        q_dim == 0 || attn_mid_dim == 0 ||
        cfg->n_activated_experts == 0 ||
        cfg->n_activated_experts > DS4_DSPARK_MAX_ACTIVATED_EXPERTS) {
        return dflash_err(err, errlen, "DSpark stage reference request has invalid input");
    }

    attn_split = calloc((size_t)mix_hc, sizeof(attn_split[0]));
    attn_x = calloc((size_t)hidden, sizeof(attn_x[0]));
    attn_norm = calloc((size_t)hidden, sizeof(attn_norm[0]));
    q_lora = calloc(cfg->q_lora_rank, sizeof(q_lora[0]));
    q_lora_norm = calloc(cfg->q_lora_rank, sizeof(q_lora_norm[0]));
    q = calloc((size_t)q_dim, sizeof(q[0]));
    wkv = calloc(cfg->head_dim, sizeof(wkv[0]));
    wkv_norm = calloc(cfg->head_dim, sizeof(wkv_norm[0]));
    attn_heads = calloc((size_t)q_dim, sizeof(attn_heads[0]));
    attn_mid = calloc((size_t)attn_mid_dim, sizeof(attn_mid[0]));
    attn_out = calloc((size_t)hidden, sizeof(attn_out[0]));
    attn_hc = calloc((size_t)hc_dim, sizeof(attn_hc[0]));
    ffn_split = calloc((size_t)mix_hc, sizeof(ffn_split[0]));
    ffn_x = calloc((size_t)hidden, sizeof(ffn_x[0]));
    ffn_norm = calloc((size_t)hidden, sizeof(ffn_norm[0]));
    shared_gate = calloc(cfg->moe_intermediate_size, sizeof(shared_gate[0]));
    shared_up = calloc(cfg->moe_intermediate_size, sizeof(shared_up[0]));
    shared_mid = calloc(cfg->moe_intermediate_size, sizeof(shared_mid[0]));
    shared_out = calloc((size_t)hidden, sizeof(shared_out[0]));
    routed_gate = calloc(cfg->moe_intermediate_size, sizeof(routed_gate[0]));
    routed_up = calloc(cfg->moe_intermediate_size, sizeof(routed_up[0]));
    routed_mid = calloc(cfg->moe_intermediate_size, sizeof(routed_mid[0]));
    routed_down = calloc((size_t)hidden, sizeof(routed_down[0]));
    routed_sum = calloc((size_t)hidden, sizeof(routed_sum[0]));
    ffn_out = calloc((size_t)hidden, sizeof(ffn_out[0]));
    if (!attn_split || !attn_x || !attn_norm || !q_lora || !q_lora_norm || !q ||
        !wkv || !wkv_norm || !attn_heads || !attn_mid || !attn_out || !attn_hc ||
        !ffn_split || !ffn_x || !ffn_norm || !shared_gate || !shared_up ||
        !shared_mid || !shared_out || !routed_gate || !routed_up ||
        !routed_mid || !routed_down || !routed_sum || !ffn_out) {
        dflash_err(err, errlen, "out of memory running DSpark stage reference");
        goto out;
    }

    if (ds4_dspark_hc_collapse_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "hc_attn_fn", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "hc_attn_scale", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "hc_attn_base", err, errlen),
            in_hc,
            cfg->hidden_size,
            cfg->hc_mult,
            cfg->hc_sinkhorn_iters,
            cfg->hc_eps,
            attn_x,
            attn_split,
            err,
            errlen) != 0 ||
        ds4_dspark_rms_norm_bf16(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn_norm.weight", err, errlen),
            attn_x,
            hidden,
            attn_norm,
            err,
            errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.wq_a.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "attn.wq_a.scale", err, errlen),
            attn_norm,
            q_lora,
            err,
            errlen) != 0 ||
        ds4_dspark_rms_norm_bf16(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.q_norm.weight", err, errlen),
            q_lora,
            cfg->q_lora_rank,
            q_lora_norm,
            err,
            errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.wq_b.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "attn.wq_b.scale", err, errlen),
            q_lora_norm,
            q,
            err,
            errlen) != 0) {
        goto out;
    }
    for (uint32_t head = 0; head < cfg->num_attention_heads; head++) {
        if (ds4_dspark_rms_norm_f32(q + (uint64_t)head * cfg->head_dim,
                                    cfg->head_dim,
                                    q + (uint64_t)head * cfg->head_dim,
                                    err,
                                    errlen) != 0) {
            goto out;
        }
    }
    if (ds4_dspark_apply_partial_rope_f32(q,
                                          cfg->num_attention_heads,
                                          cfg->head_dim,
                                          cfg->qk_rope_head_dim,
                                          position,
                                          cfg->compress_rope_theta,
                                          cfg->rope_original_max_position_embeddings,
                                          cfg->rope_factor,
                                          cfg->rope_beta_fast,
                                          cfg->rope_beta_slow,
                                          false,
                                          err,
                                          errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.wkv.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "attn.wkv.scale", err, errlen),
            attn_norm,
            wkv,
            err,
            errlen) != 0 ||
        ds4_dspark_rms_norm_bf16(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.kv_norm.weight", err, errlen),
            wkv,
            cfg->head_dim,
            wkv_norm,
            err,
            errlen) != 0 ||
        ds4_dspark_apply_partial_rope_f32(wkv_norm,
                                          1,
                                          cfg->head_dim,
                                          cfg->qk_rope_head_dim,
                                          position,
                                          cfg->compress_rope_theta,
                                          cfg->rope_original_max_position_embeddings,
                                          cfg->rope_factor,
                                          cfg->rope_beta_fast,
                                          cfg->rope_beta_slow,
                                          false,
                                          err,
                                          errlen) != 0 ||
        ds4_dspark_sparse_attention_one_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.attn_sink", err, errlen),
            q,
            wkv_norm,
            cfg->num_attention_heads,
            cfg->head_dim,
            attn_heads,
            err,
            errlen) != 0 ||
        ds4_dspark_apply_partial_rope_f32(attn_heads,
                                          cfg->num_attention_heads,
                                          cfg->head_dim,
                                          cfg->qk_rope_head_dim,
                                          position,
                                          cfg->compress_rope_theta,
                                          cfg->rope_original_max_position_embeddings,
                                          cfg->rope_factor,
                                          cfg->rope_beta_fast,
                                          cfg->rope_beta_slow,
                                          true,
                                          err,
                                          errlen) != 0 ||
        ds4_dspark_grouped_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.wo_a.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "attn.wo_a.scale", err, errlen),
            attn_heads,
            cfg->o_groups,
            attn_mid,
            err,
            errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "attn.wo_b.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "attn.wo_b.scale", err, errlen),
            attn_mid,
            attn_out,
            err,
            errlen) != 0 ||
        ds4_dspark_hc_expand_f32(attn_out,
                                 in_hc,
                                 attn_split,
                                 cfg->hidden_size,
                                 cfg->hc_mult,
                                 attn_hc,
                                 err,
                                 errlen) != 0 ||
        ds4_dspark_hc_collapse_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "hc_ffn_fn", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "hc_ffn_scale", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "hc_ffn_base", err, errlen),
            attn_hc,
            cfg->hidden_size,
            cfg->hc_mult,
            cfg->hc_sinkhorn_iters,
            cfg->hc_eps,
            ffn_x,
            ffn_split,
            err,
            errlen) != 0 ||
        ds4_dspark_rms_norm_bf16(
            w,
            dspark_find_stage_tensor_required(w, stage, "ffn_norm.weight", err, errlen),
            ffn_x,
            hidden,
            ffn_norm,
            err,
            errlen) != 0 ||
        ds4_dspark_moe_gate_topk_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "ffn.gate.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "ffn.gate.bias", err, errlen),
            ffn_norm,
            cfg->n_activated_experts,
            cfg->routed_scaling_factor,
            route_indices,
            route_weights,
            err,
            errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "ffn.shared_experts.w1.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "ffn.shared_experts.w1.scale", err, errlen),
            ffn_norm,
            shared_gate,
            err,
            errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "ffn.shared_experts.w3.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "ffn.shared_experts.w3.scale", err, errlen),
            ffn_norm,
            shared_up,
            err,
            errlen) != 0 ||
        ds4_dspark_swiglu_f32(shared_gate,
                              shared_up,
                              cfg->moe_intermediate_size,
                              cfg->swiglu_limit,
                              shared_mid,
                              err,
                              errlen) != 0 ||
        ds4_dspark_linear_f8_f32(
            w,
            dspark_find_stage_tensor_required(w, stage, "ffn.shared_experts.w2.weight", err, errlen),
            dspark_find_stage_tensor_required(w, stage, "ffn.shared_experts.w2.scale", err, errlen),
            shared_mid,
            shared_out,
            err,
            errlen) != 0) {
        goto out;
    }

    for (uint32_t slot = 0; slot < cfg->n_activated_experts; slot++) {
        char w1_name[DS4_DFLASH_MAX_TENSOR_NAME];
        char w1_scale_name[DS4_DFLASH_MAX_TENSOR_NAME];
        char w2_name[DS4_DFLASH_MAX_TENSOR_NAME];
        char w2_scale_name[DS4_DFLASH_MAX_TENSOR_NAME];
        char w3_name[DS4_DFLASH_MAX_TENSOR_NAME];
        char w3_scale_name[DS4_DFLASH_MAX_TENSOR_NAME];
        const uint32_t expert = route_indices[slot];
        if (dspark_stage_expert_tensor_name(w1_name, sizeof(w1_name), stage, expert, "w1.weight", err, errlen) != 0 ||
            dspark_stage_expert_tensor_name(w1_scale_name, sizeof(w1_scale_name), stage, expert, "w1.scale", err, errlen) != 0 ||
            dspark_stage_expert_tensor_name(w2_name, sizeof(w2_name), stage, expert, "w2.weight", err, errlen) != 0 ||
            dspark_stage_expert_tensor_name(w2_scale_name, sizeof(w2_scale_name), stage, expert, "w2.scale", err, errlen) != 0 ||
            dspark_stage_expert_tensor_name(w3_name, sizeof(w3_name), stage, expert, "w3.weight", err, errlen) != 0 ||
            dspark_stage_expert_tensor_name(w3_scale_name, sizeof(w3_scale_name), stage, expert, "w3.scale", err, errlen) != 0 ||
            ds4_dspark_linear_fp4_f32(
                w,
                ds4_dspark_weights_find_tensor(w, w1_name),
                ds4_dspark_weights_find_tensor(w, w1_scale_name),
                ffn_norm,
                routed_gate,
                err,
                errlen) != 0 ||
            ds4_dspark_linear_fp4_f32(
                w,
                ds4_dspark_weights_find_tensor(w, w3_name),
                ds4_dspark_weights_find_tensor(w, w3_scale_name),
                ffn_norm,
                routed_up,
                err,
                errlen) != 0 ||
            ds4_dspark_swiglu_f32(routed_gate,
                                  routed_up,
                                  cfg->moe_intermediate_size,
                                  cfg->swiglu_limit,
                                  routed_mid,
                                  err,
                                  errlen) != 0) {
            goto out;
        }
        for (uint32_t i = 0; i < cfg->moe_intermediate_size; i++) {
            routed_mid[i] *= route_weights[slot];
        }
        if (ds4_dspark_linear_fp4_f32(
                w,
                ds4_dspark_weights_find_tensor(w, w2_name),
                ds4_dspark_weights_find_tensor(w, w2_scale_name),
                routed_mid,
                routed_down,
                err,
                errlen) != 0) {
            goto out;
        }
        for (uint32_t i = 0; i < cfg->hidden_size; i++) {
            routed_sum[i] += routed_down[i];
        }
    }

    for (uint32_t i = 0; i < cfg->hidden_size; i++) {
        ffn_out[i] = shared_out[i] + routed_sum[i];
    }
    if (ds4_dspark_hc_expand_f32(ffn_out,
                                 attn_hc,
                                 ffn_split,
                                 cfg->hidden_size,
                                 cfg->hc_mult,
                                 out_hc,
                                 err,
                                 errlen) != 0) {
        goto out;
    }

    if (stats) {
        stats->stage = stage;
        stats->n_routes = cfg->n_activated_experts;
        for (uint32_t i = 0; i < cfg->n_activated_experts; i++) {
            stats->route_indices[i] = route_indices[i];
            stats->route_weights[i] = route_weights[i];
        }
        stats->attn_hc_checksum = dspark_checksum_f32(attn_hc, hc_dim);
        stats->ffn_hc_checksum = dspark_checksum_f32(out_hc, hc_dim);
        stats->ffn_out_checksum = dspark_checksum_f32(ffn_out, hidden);
        stats->routed_sum_checksum = dspark_checksum_f32(routed_sum, hidden);
        stats->shared_out_checksum = dspark_checksum_f32(shared_out, hidden);
        for (uint32_t i = 0; i < 4u && i < hidden; i++) {
            stats->ffn_hc_first[i] = out_hc[i];
        }
    }
    rc = 0;

out:
    free(attn_split);
    free(attn_x);
    free(attn_norm);
    free(q_lora);
    free(q_lora_norm);
    free(q);
    free(wkv);
    free(wkv_norm);
    free(attn_heads);
    free(attn_mid);
    free(attn_out);
    free(attn_hc);
    free(ffn_split);
    free(ffn_x);
    free(ffn_norm);
    free(shared_gate);
    free(shared_up);
    free(shared_mid);
    free(shared_out);
    free(routed_gate);
    free(routed_up);
    free(routed_mid);
    free(routed_down);
    free(routed_sum);
    free(ffn_out);
    return rc;
}

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
                                                 size_t errlen) {
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t hc_mult = cfg ? cfg->hc_mult : 0;
    const uint64_t hc_dim = hidden * hc_mult;
    const uint64_t mix_hc = (hc_mult + 2u) * hc_mult;
    const uint64_t q_dim = cfg ? (uint64_t)cfg->num_attention_heads * cfg->head_dim : 0;
    const uint64_t attn_mid_dim = cfg ? (uint64_t)cfg->o_groups * cfg->o_lora_rank : 0;
    const uint64_t total_kv_rows = (uint64_t)context_rows + n_rows;
    const bool block_timing = getenv("DS4_DSPARK_BLOCK_TIMING") != NULL;
    const double t0 = block_timing ? dflash_now_sec() : 0.0;
    double t_bound = 0.0;
    double t_alloc = 0.0;
    double t_qkv = 0.0;
    double t_sparse = 0.0;
    double t_attn_out = 0.0;
    dspark_stage_block_tensors tensors;
    float *attn_split_rows = NULL;
    float *q_rows = NULL;
    float *wkv_rows = NULL;
    float *query_wkv_rows = NULL;
    float *attn_heads_rows = NULL;
    float *attn_hc_rows = NULL;
    float *attn_x = NULL;
    float *attn_norm = NULL;
    float *q_lora = NULL;
    float *q_lora_norm = NULL;
    float *attn_mid = NULL;
    float *attn_out = NULL;
    float *ffn_split = NULL;
    float *ffn_x = NULL;
    float *ffn_norm = NULL;
    uint32_t route_indices[DS4_DSPARK_MAX_ACTIVATED_EXPERTS];
    float route_weights[DS4_DSPARK_MAX_ACTIVATED_EXPERTS];
    float *shared_gate = NULL;
    float *shared_up = NULL;
    float *shared_mid = NULL;
    float *shared_out = NULL;
    float *routed_gate = NULL;
    float *routed_up = NULL;
    float *routed_mid = NULL;
    float *routed_down = NULL;
    float *routed_sum = NULL;
    float *ffn_out = NULL;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !in_hc_rows || !out_hc_rows ||
        stage >= cfg->n_mtp_stages || n_rows == 0 || n_rows > cfg->block_size ||
        hidden == 0 || hc_mult == 0 || hc_dim == 0 || mix_hc == 0 ||
        q_dim == 0 || attn_mid_dim == 0 ||
        cfg->n_activated_experts == 0 ||
        cfg->n_activated_experts > DS4_DSPARK_MAX_ACTIVATED_EXPERTS) {
        return dflash_err(err, errlen, "DSpark stage block reference request has invalid input");
    }
    if (context_rows > 0 && (!context_main_rows || !context_positions)) {
        return dflash_err(err, errlen, "DSpark stage block context is missing");
    }
    if ((uint64_t)n_rows * hc_dim > SIZE_MAX / sizeof(float) ||
        (uint64_t)n_rows * mix_hc > SIZE_MAX / sizeof(float) ||
        (uint64_t)n_rows * q_dim > SIZE_MAX / sizeof(float) ||
        total_kv_rows > SIZE_MAX / sizeof(float) / cfg->head_dim) {
        return dflash_err(err, errlen, "DSpark stage block buffers are too large");
    }
    if (dspark_stage_block_tensors_bind(&tensors, w, stage, err, errlen) != 0) {
        return 1;
    }
    if (block_timing) t_bound = dflash_now_sec();

    attn_split_rows = calloc((size_t)((uint64_t)n_rows * mix_hc), sizeof(attn_split_rows[0]));
    q_rows = calloc((size_t)((uint64_t)n_rows * q_dim), sizeof(q_rows[0]));
    wkv_rows = calloc((size_t)(total_kv_rows * cfg->head_dim), sizeof(wkv_rows[0]));
    attn_heads_rows = calloc((size_t)((uint64_t)n_rows * q_dim), sizeof(attn_heads_rows[0]));
    attn_hc_rows = calloc((size_t)((uint64_t)n_rows * hc_dim), sizeof(attn_hc_rows[0]));
    attn_x = calloc((size_t)hidden, sizeof(attn_x[0]));
    attn_norm = calloc((size_t)hidden, sizeof(attn_norm[0]));
    q_lora = calloc(cfg->q_lora_rank, sizeof(q_lora[0]));
    q_lora_norm = calloc(cfg->q_lora_rank, sizeof(q_lora_norm[0]));
    attn_mid = calloc((size_t)attn_mid_dim, sizeof(attn_mid[0]));
    attn_out = calloc((size_t)hidden, sizeof(attn_out[0]));
    ffn_split = calloc((size_t)mix_hc, sizeof(ffn_split[0]));
    ffn_x = calloc((size_t)hidden, sizeof(ffn_x[0]));
    ffn_norm = calloc((size_t)hidden, sizeof(ffn_norm[0]));
    shared_gate = calloc(cfg->moe_intermediate_size, sizeof(shared_gate[0]));
    shared_up = calloc(cfg->moe_intermediate_size, sizeof(shared_up[0]));
    shared_mid = calloc(cfg->moe_intermediate_size, sizeof(shared_mid[0]));
    shared_out = calloc((size_t)hidden, sizeof(shared_out[0]));
    routed_gate = calloc(cfg->moe_intermediate_size, sizeof(routed_gate[0]));
    routed_up = calloc(cfg->moe_intermediate_size, sizeof(routed_up[0]));
    routed_mid = calloc(cfg->moe_intermediate_size, sizeof(routed_mid[0]));
    routed_down = calloc((size_t)hidden, sizeof(routed_down[0]));
    routed_sum = calloc((size_t)hidden, sizeof(routed_sum[0]));
    ffn_out = calloc((size_t)hidden, sizeof(ffn_out[0]));
    if (!attn_split_rows || !q_rows || !wkv_rows || !attn_heads_rows ||
        !attn_hc_rows || !attn_x || !attn_norm || !q_lora || !q_lora_norm ||
        !attn_mid || !attn_out || !ffn_split || !ffn_x || !ffn_norm ||
        !shared_gate || !shared_up || !shared_mid || !shared_out ||
        !routed_gate || !routed_up || !routed_mid || !routed_down ||
        !routed_sum || !ffn_out) {
        dflash_err(err, errlen, "out of memory running DSpark stage block reference");
        goto out;
    }
    if (block_timing) t_alloc = dflash_now_sec();
    query_wkv_rows = wkv_rows + (uint64_t)context_rows * cfg->head_dim;

    for (uint32_t row = 0; row < context_rows; row++) {
        float *wkv = wkv_rows + (uint64_t)row * cfg->head_dim;
        if (ds4_dspark_linear_f8_f32(
                w,
                tensors.attn_wkv_weight,
                tensors.attn_wkv_scale,
                context_main_rows + (uint64_t)row * hidden,
                wkv,
                err,
                errlen) != 0 ||
            ds4_dspark_rms_norm_bf16(
                w,
                tensors.attn_kv_norm_weight,
                wkv,
                cfg->head_dim,
                wkv,
                err,
                errlen) != 0 ||
            ds4_dspark_apply_partial_rope_f32(wkv,
                                              1,
                                              cfg->head_dim,
                                              cfg->qk_rope_head_dim,
                                              context_positions[row],
                                              cfg->compress_rope_theta,
                                              cfg->rope_original_max_position_embeddings,
                                              cfg->rope_factor,
                                              cfg->rope_beta_fast,
                                              cfg->rope_beta_slow,
                                              false,
                                              err,
                                              errlen) != 0) {
            goto out;
        }
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        const float *in_hc = in_hc_rows + (uint64_t)row * hc_dim;
        float *attn_split = attn_split_rows + (uint64_t)row * mix_hc;
        float *q = q_rows + (uint64_t)row * q_dim;
        float *wkv = query_wkv_rows + (uint64_t)row * cfg->head_dim;
        if (ds4_dspark_hc_collapse_f32(
                w,
                tensors.hc_attn_fn,
                tensors.hc_attn_scale,
                tensors.hc_attn_base,
                in_hc,
                cfg->hidden_size,
                cfg->hc_mult,
                cfg->hc_sinkhorn_iters,
                cfg->hc_eps,
                attn_x,
                attn_split,
                err,
                errlen) != 0 ||
            ds4_dspark_rms_norm_bf16(
                w,
                tensors.attn_norm_weight,
                attn_x,
                hidden,
                attn_norm,
                err,
                errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.attn_wq_a_weight,
                tensors.attn_wq_a_scale,
                attn_norm,
                q_lora,
                err,
                errlen) != 0 ||
            ds4_dspark_rms_norm_bf16(
                w,
                tensors.attn_q_norm_weight,
                q_lora,
                cfg->q_lora_rank,
                q_lora_norm,
                err,
                errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.attn_wq_b_weight,
                tensors.attn_wq_b_scale,
                q_lora_norm,
                q,
                err,
                errlen) != 0) {
            goto out;
        }
        for (uint32_t head = 0; head < cfg->num_attention_heads; head++) {
            if (ds4_dspark_rms_norm_f32(q + (uint64_t)head * cfg->head_dim,
                                        cfg->head_dim,
                                        q + (uint64_t)head * cfg->head_dim,
                                        err,
                                        errlen) != 0) {
                goto out;
            }
        }
        if (ds4_dspark_apply_partial_rope_f32(q,
                                              cfg->num_attention_heads,
                                              cfg->head_dim,
                                              cfg->qk_rope_head_dim,
                                              position0 + row,
                                              cfg->compress_rope_theta,
                                              cfg->rope_original_max_position_embeddings,
                                              cfg->rope_factor,
                                              cfg->rope_beta_fast,
                                              cfg->rope_beta_slow,
                                              false,
                                              err,
                                              errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.attn_wkv_weight,
                tensors.attn_wkv_scale,
                attn_norm,
                wkv,
                err,
                errlen) != 0 ||
            ds4_dspark_rms_norm_bf16(
                w,
                tensors.attn_kv_norm_weight,
                wkv,
                cfg->head_dim,
                wkv,
                err,
                errlen) != 0 ||
            ds4_dspark_apply_partial_rope_f32(wkv,
                                              1,
                                              cfg->head_dim,
                                              cfg->qk_rope_head_dim,
                                              position0 + row,
                                              cfg->compress_rope_theta,
                                              cfg->rope_original_max_position_embeddings,
                                              cfg->rope_factor,
                                              cfg->rope_beta_fast,
                                              cfg->rope_beta_slow,
                                              false,
                                              err,
                                              errlen) != 0) {
            goto out;
        }
    }
    if (block_timing) t_qkv = dflash_now_sec();

    if (ds4_dspark_sparse_attention_block_f32(
            w,
            tensors.attn_sink,
            q_rows,
            wkv_rows,
            n_rows,
            (uint32_t)total_kv_rows,
            cfg->num_attention_heads,
            cfg->head_dim,
            attn_heads_rows,
            err,
            errlen) != 0) {
        goto out;
    }
    if (block_timing) t_sparse = dflash_now_sec();

    for (uint32_t row = 0; row < n_rows; row++) {
        const float *in_hc = in_hc_rows + (uint64_t)row * hc_dim;
        const float *attn_split = attn_split_rows + (uint64_t)row * mix_hc;
        float *attn_heads = attn_heads_rows + (uint64_t)row * q_dim;
        float *attn_hc = attn_hc_rows + (uint64_t)row * hc_dim;
        if (ds4_dspark_apply_partial_rope_f32(attn_heads,
                                              cfg->num_attention_heads,
                                              cfg->head_dim,
                                              cfg->qk_rope_head_dim,
                                              position0 + row,
                                              cfg->compress_rope_theta,
                                              cfg->rope_original_max_position_embeddings,
                                              cfg->rope_factor,
                                              cfg->rope_beta_fast,
                                              cfg->rope_beta_slow,
                                              true,
                                              err,
                                              errlen) != 0 ||
            ds4_dspark_grouped_linear_f8_f32(
                w,
                tensors.attn_wo_a_weight,
                tensors.attn_wo_a_scale,
                attn_heads,
                cfg->o_groups,
                attn_mid,
                err,
                errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.attn_wo_b_weight,
                tensors.attn_wo_b_scale,
                attn_mid,
                attn_out,
                err,
                errlen) != 0 ||
            ds4_dspark_hc_expand_f32(attn_out,
                                     in_hc,
                                     attn_split,
                                     cfg->hidden_size,
                                     cfg->hc_mult,
                                     attn_hc,
                                     err,
                                     errlen) != 0) {
            goto out;
        }
    }
    if (block_timing) t_attn_out = dflash_now_sec();

    for (uint32_t row = 0; row < n_rows; row++) {
        const float *attn_hc = attn_hc_rows + (uint64_t)row * hc_dim;
        float *out_hc = out_hc_rows + (uint64_t)row * hc_dim;
        if (ds4_dspark_hc_collapse_f32(
                w,
                tensors.hc_ffn_fn,
                tensors.hc_ffn_scale,
                tensors.hc_ffn_base,
                attn_hc,
                cfg->hidden_size,
                cfg->hc_mult,
                cfg->hc_sinkhorn_iters,
                cfg->hc_eps,
                ffn_x,
                ffn_split,
                err,
                errlen) != 0 ||
            ds4_dspark_rms_norm_bf16(
                w,
                tensors.ffn_norm_weight,
                ffn_x,
                hidden,
                ffn_norm,
                err,
                errlen) != 0 ||
            ds4_dspark_moe_gate_topk_f32(
                w,
                tensors.ffn_gate_weight,
                tensors.ffn_gate_bias,
                ffn_norm,
                cfg->n_activated_experts,
                cfg->routed_scaling_factor,
                route_indices,
                route_weights,
                err,
                errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.shared_w1_weight,
                tensors.shared_w1_scale,
                ffn_norm,
                shared_gate,
                err,
                errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.shared_w3_weight,
                tensors.shared_w3_scale,
                ffn_norm,
                shared_up,
                err,
                errlen) != 0 ||
            ds4_dspark_swiglu_f32(shared_gate,
                                  shared_up,
                                  cfg->moe_intermediate_size,
                                  cfg->swiglu_limit,
                                  shared_mid,
                                  err,
                                  errlen) != 0 ||
            ds4_dspark_linear_f8_f32(
                w,
                tensors.shared_w2_weight,
                tensors.shared_w2_scale,
                shared_mid,
                shared_out,
                err,
                errlen) != 0) {
            goto out;
        }

        memset(routed_sum, 0, (size_t)hidden * sizeof(routed_sum[0]));
        for (uint32_t slot = 0; slot < cfg->n_activated_experts; slot++) {
            char w1_name[DS4_DFLASH_MAX_TENSOR_NAME];
            char w1_scale_name[DS4_DFLASH_MAX_TENSOR_NAME];
            char w2_name[DS4_DFLASH_MAX_TENSOR_NAME];
            char w2_scale_name[DS4_DFLASH_MAX_TENSOR_NAME];
            char w3_name[DS4_DFLASH_MAX_TENSOR_NAME];
            char w3_scale_name[DS4_DFLASH_MAX_TENSOR_NAME];
            const uint32_t expert = route_indices[slot];
            if (dspark_stage_expert_tensor_name(w1_name, sizeof(w1_name), stage, expert, "w1.weight", err, errlen) != 0 ||
                dspark_stage_expert_tensor_name(w1_scale_name, sizeof(w1_scale_name), stage, expert, "w1.scale", err, errlen) != 0 ||
                dspark_stage_expert_tensor_name(w2_name, sizeof(w2_name), stage, expert, "w2.weight", err, errlen) != 0 ||
                dspark_stage_expert_tensor_name(w2_scale_name, sizeof(w2_scale_name), stage, expert, "w2.scale", err, errlen) != 0 ||
                dspark_stage_expert_tensor_name(w3_name, sizeof(w3_name), stage, expert, "w3.weight", err, errlen) != 0 ||
                dspark_stage_expert_tensor_name(w3_scale_name, sizeof(w3_scale_name), stage, expert, "w3.scale", err, errlen) != 0 ||
                ds4_dspark_linear_fp4_f32(
                    w,
                    ds4_dspark_weights_find_tensor(w, w1_name),
                    ds4_dspark_weights_find_tensor(w, w1_scale_name),
                    ffn_norm,
                    routed_gate,
                    err,
                    errlen) != 0 ||
                ds4_dspark_linear_fp4_f32(
                    w,
                    ds4_dspark_weights_find_tensor(w, w3_name),
                    ds4_dspark_weights_find_tensor(w, w3_scale_name),
                    ffn_norm,
                    routed_up,
                    err,
                    errlen) != 0 ||
                ds4_dspark_swiglu_f32(routed_gate,
                                      routed_up,
                                      cfg->moe_intermediate_size,
                                      cfg->swiglu_limit,
                                      routed_mid,
                                      err,
                                      errlen) != 0) {
                goto out;
            }
            for (uint32_t i = 0; i < cfg->moe_intermediate_size; i++) {
                routed_mid[i] *= route_weights[slot];
            }
            if (ds4_dspark_linear_fp4_f32(
                    w,
                    ds4_dspark_weights_find_tensor(w, w2_name),
                    ds4_dspark_weights_find_tensor(w, w2_scale_name),
                    routed_mid,
                    routed_down,
                    err,
                    errlen) != 0) {
                goto out;
            }
            for (uint32_t i = 0; i < cfg->hidden_size; i++) {
                routed_sum[i] += routed_down[i];
            }
        }

        for (uint32_t i = 0; i < cfg->hidden_size; i++) {
            ffn_out[i] = shared_out[i] + routed_sum[i];
        }
        if (ds4_dspark_hc_expand_f32(ffn_out,
                                     attn_hc,
                                     ffn_split,
                                     cfg->hidden_size,
                                     cfg->hc_mult,
                                     out_hc,
                                     err,
                                     errlen) != 0) {
            goto out;
        }
    }

    if (block_timing) {
        const double t_done = dflash_now_sec();
        fprintf(stderr,
                "ds4: dspark block timing stage=%u rows=%u bind=%.3f ms alloc=%.3f ms qkv=%.3f ms sparse_attn=%.3f ms attn_out=%.3f ms ffn=%.3f ms total=%.3f ms\n",
                stage,
                n_rows,
                (t_bound - t0) * 1000.0,
                (t_alloc - t_bound) * 1000.0,
                (t_qkv - t_alloc) * 1000.0,
                (t_sparse - t_qkv) * 1000.0,
                (t_attn_out - t_sparse) * 1000.0,
                (t_done - t_attn_out) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    rc = 0;

out:
    free(attn_split_rows);
    free(q_rows);
    free(wkv_rows);
    free(attn_heads_rows);
    free(attn_hc_rows);
    free(attn_x);
    free(attn_norm);
    free(q_lora);
    free(q_lora_norm);
    free(attn_mid);
    free(attn_out);
    free(ffn_split);
    free(ffn_x);
    free(ffn_norm);
    free(shared_gate);
    free(shared_up);
    free(shared_mid);
    free(shared_out);
    free(routed_gate);
    free(routed_up);
    free(routed_mid);
    free(routed_down);
    free(routed_sum);
    free(ffn_out);
    return rc;
}

int ds4_dspark_run_stage_block_reference(const ds4_dspark_weights *w,
                                         const ds4_dspark_config *cfg,
                                         uint32_t stage,
                                         uint64_t position0,
                                         const float *in_hc_rows,
                                         uint32_t n_rows,
                                         float *out_hc_rows,
                                         char *err,
                                         size_t errlen) {
    return ds4_dspark_run_stage_block_context_reference(w,
                                                        cfg,
                                                        stage,
                                                        NULL,
                                                        NULL,
                                                        0,
                                                        position0,
                                                        in_hc_rows,
                                                        n_rows,
                                                        out_hc_rows,
                                                        err,
                                                        errlen);
}

int ds4_dspark_init_hc_block_from_main_f32(const ds4_dspark_config *cfg,
                                           const float *main_x,
                                           uint32_t n_rows,
                                           float *out_hc_rows,
                                           char *err,
                                           size_t errlen) {
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t hc_mult = cfg ? cfg->hc_mult : 0;
    const uint64_t hc_dim = hidden * hc_mult;

    if (!cfg || !cfg->loaded || !main_x || !out_hc_rows ||
        n_rows == 0 || n_rows > cfg->block_size ||
        hidden == 0 || hc_mult == 0 || hc_dim == 0) {
        return dflash_err(err, errlen, "invalid DSpark HC block initialization request");
    }
    if ((uint64_t)n_rows * hc_dim > SIZE_MAX / sizeof(out_hc_rows[0])) {
        return dflash_err(err, errlen, "DSpark HC block initialization buffer is too large");
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        float *row_hc = out_hc_rows + (uint64_t)row * hc_dim;
        for (uint32_t h = 0; h < cfg->hc_mult; h++) {
            memcpy(row_hc + (uint64_t)h * hidden,
                   main_x,
                   (size_t)hidden * sizeof(main_x[0]));
        }
    }
    return 0;
}

int ds4_dspark_project_main_hidden(const ds4_dspark_weights *w,
                                   const ds4_dspark_config *cfg,
                                   const float *main_hidden,
                                   float *main_x,
                                   char *err,
                                   size_t errlen) {
    const ds4_dflash_tensor *main_proj = NULL;
    const ds4_dflash_tensor *main_scale = NULL;
    const ds4_dflash_tensor *main_norm = NULL;
    uint64_t hidden = 0;
    uint64_t in_dim = 0;
    uint64_t scale_rows = 0;
    uint64_t scale_cols = 0;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !main_hidden || !main_x) {
        return dflash_err(err, errlen, "invalid DSpark main hidden projection request");
    }
    hidden = cfg->hidden_size;
    if (hidden == 0 || cfg->n_target_layer_ids == 0 ||
        hidden > UINT64_MAX / cfg->n_target_layer_ids) {
        return dflash_err(err, errlen, "DSpark main hidden projection dimensions are invalid");
    }
    in_dim = hidden * cfg->n_target_layer_ids;
    scale_rows = ceil_div_u64(hidden, 128u);
    scale_cols = ceil_div_u64(in_dim, 128u);

    main_proj = ds4_dspark_weights_find_tensor(w, "mtp.0.main_proj.weight");
    main_scale = ds4_dspark_weights_find_tensor(w, "mtp.0.main_proj.scale");
    main_norm = ds4_dspark_weights_find_tensor(w, "mtp.0.main_norm.weight");
    if (!main_proj || !main_scale || !main_norm) {
        return dflash_err(err, errlen, "DSpark main projection tensors are not bound");
    }
    if (main_proj->dtype != DS4_DFLASH_TENSOR_F8_E4M3 ||
        main_proj->ndim != 2 ||
        main_proj->shape[0] != hidden ||
        main_proj->shape[1] != in_dim ||
        main_scale->dtype != DS4_DFLASH_TENSOR_F8_E8M0 ||
        main_scale->ndim != 2 ||
        main_scale->shape[0] != scale_rows ||
        main_scale->shape[1] != scale_cols ||
        main_norm->dtype != DS4_DFLASH_TENSOR_BF16 ||
        main_norm->ndim != 1 ||
        main_norm->shape[0] != hidden) {
        return dflash_err(err, errlen,
                          "DSpark main projection tensor layout does not match config");
    }
    if (ds4_dspark_linear_f8_f32(w,
                                 main_proj,
                                 main_scale,
                                 main_hidden,
                                 main_x,
                                 err,
                                 errlen) != 0 ||
        ds4_dspark_rms_norm_bf16(w,
                                 main_norm,
                                 main_x,
                                 hidden,
                                 main_x,
                                 err,
                                 errlen) != 0) {
        return 1;
    }
    return 0;
}

int ds4_dspark_final_hc_head_f32(const ds4_dspark_weights *w,
                                 const ds4_dspark_config *cfg,
                                 const float *in_hc,
                                 float *out,
                                 char *err,
                                 size_t errlen) {
    const ds4_dflash_tensor *fn = NULL;
    const ds4_dflash_tensor *base = NULL;
    const ds4_dflash_tensor *scale = NULL;
    const unsigned char *fn_data = NULL;
    const unsigned char *base_data = NULL;
    const unsigned char *scale_data = NULL;
    float *flat = NULL;
    float *weights = NULL;
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t hc_mult = cfg ? cfg->hc_mult : 0;
    const uint64_t hc_dim = hidden * hc_mult;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !in_hc || !out ||
        hidden == 0 || hc_mult == 0 || hc_dim == 0) {
        return dflash_err(err, errlen, "DSpark final HC head request has invalid input");
    }
    fn = ds4_dspark_weights_find_tensor(w, "mtp.2.hc_head_fn");
    base = ds4_dspark_weights_find_tensor(w, "mtp.2.hc_head_base");
    scale = ds4_dspark_weights_find_tensor(w, "mtp.2.hc_head_scale");
    if (!fn || !base || !scale ||
        fn->dtype != DS4_DFLASH_TENSOR_F32 ||
        fn->ndim != 2 ||
        fn->shape[0] != hc_mult ||
        fn->shape[1] != hc_dim ||
        base->dtype != DS4_DFLASH_TENSOR_F32 ||
        base->ndim != 1 ||
        base->shape[0] != hc_mult ||
        scale->dtype != DS4_DFLASH_TENSOR_F32 ||
        scale->ndim != 1 ||
        scale->shape[0] != 1) {
        return dflash_err(err, errlen, "DSpark final HC head tensor layout is invalid");
    }
    flat = calloc((size_t)hc_dim, sizeof(flat[0]));
    weights = calloc((size_t)hc_mult, sizeof(weights[0]));
    if (!flat || !weights) {
        dflash_err(err, errlen, "out of memory running DSpark final HC head");
        goto out;
    }
    if (ds4_dspark_rms_norm_f32(in_hc, hc_dim, flat, err, errlen) != 0 ||
        dspark_tensor_data_ptr(w, fn, 0, fn->nbytes, &fn_data, err, errlen) != 0 ||
        dspark_tensor_data_ptr(w, base, 0, base->nbytes, &base_data, err, errlen) != 0 ||
        dspark_tensor_data_ptr(w, scale, 0, scale->nbytes, &scale_data, err, errlen) != 0) {
        goto out;
    }

    const float s = read_le_f32(scale_data);
    for (uint32_t h = 0; h < cfg->hc_mult; h++) {
        double dot = 0.0;
        const uint64_t row = (uint64_t)h * hc_dim;
        for (uint64_t i = 0; i < hc_dim; i++) {
            dot += (double)read_le_f32(fn_data + (row + i) * 4u) * (double)flat[i];
        }
        weights[h] = dspark_sigmoidf((float)dot * s +
                                     read_le_f32(base_data + (uint64_t)h * 4u)) +
                     cfg->hc_eps;
    }
    for (uint32_t d = 0; d < cfg->hidden_size; d++) {
        double sum = 0.0;
        for (uint32_t h = 0; h < cfg->hc_mult; h++) {
            sum += (double)weights[h] * (double)in_hc[(uint64_t)h * cfg->hidden_size + d];
        }
        out[d] = (float)sum;
    }
    rc = 0;

out:
    free(weights);
    free(flat);
    return rc;
}

int ds4_dspark_final_norm_f32(const ds4_dspark_weights *w,
                              const ds4_dspark_config *cfg,
                              const float *in,
                              float *out,
                              char *err,
                              size_t errlen) {
    const ds4_dflash_tensor *norm = NULL;
    if (!w || !w->loaded || !cfg || !cfg->loaded || !in || !out ||
        cfg->hidden_size == 0) {
        return dflash_err(err, errlen, "DSpark final norm request has invalid input");
    }
    norm = ds4_dspark_weights_find_tensor(w, "mtp.2.norm.weight");
    if (!norm) {
        return dflash_err(err, errlen, "DSpark final norm tensor is not bound");
    }
    return ds4_dspark_rms_norm_bf16(w,
                                    norm,
                                    in,
                                    cfg->hidden_size,
                                    out,
                                    err,
                                    errlen);
}

int ds4_dspark_final_block_norm_f32(const ds4_dspark_weights *w,
                                    const ds4_dspark_config *cfg,
                                    const float *in_hc_rows,
                                    uint32_t n_rows,
                                    float *hidden_rows,
                                    float *normed_rows,
                                    char *err,
                                    size_t errlen) {
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t hc_dim = cfg ? (uint64_t)cfg->hidden_size * cfg->hc_mult : 0;
    float *scratch_hidden = NULL;
    int rc = 1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !in_hc_rows ||
        n_rows == 0 || n_rows > cfg->block_size || hidden == 0 ||
        hc_dim == 0 || !normed_rows) {
        return dflash_err(err, errlen, "DSpark final block norm request has invalid input");
    }
    if ((uint64_t)n_rows * hidden > SIZE_MAX / sizeof(float) ||
        (uint64_t)n_rows * hc_dim > SIZE_MAX / sizeof(float)) {
        return dflash_err(err, errlen, "DSpark final block norm buffers are too large");
    }
    if (!hidden_rows) {
        scratch_hidden = calloc((size_t)hidden, sizeof(scratch_hidden[0]));
        if (!scratch_hidden) {
            return dflash_err(err, errlen, "out of memory running DSpark final block norm");
        }
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        float *hidden_out = hidden_rows ?
            hidden_rows + (uint64_t)row * hidden : scratch_hidden;
        if (ds4_dspark_final_hc_head_f32(w,
                                         cfg,
                                         in_hc_rows + (uint64_t)row * hc_dim,
                                         hidden_out,
                                         err,
                                         errlen) != 0 ||
            ds4_dspark_final_norm_f32(w,
                                      cfg,
                                      hidden_out,
                                      normed_rows + (uint64_t)row * hidden,
                                      err,
                                      errlen) != 0) {
            goto out;
        }
    }

    rc = 0;
out:
    free(scratch_hidden);
    return rc;
}

int ds4_dspark_markov_prev_embedding_f32(const ds4_dspark_weights *w,
                                         const ds4_dspark_config *cfg,
                                         uint32_t prev_token_id,
                                         float *embedding,
                                         char *err,
                                         size_t errlen) {
    const ds4_dflash_tensor *markov_w1 = NULL;
    if (!w || !w->loaded || !cfg || !cfg->loaded || !embedding ||
        cfg->markov_rank == 0 || prev_token_id >= cfg->vocab_size) {
        return dflash_err(err, errlen, "DSpark Markov embedding request has invalid input");
    }
    markov_w1 = ds4_dspark_weights_find_tensor(w, "mtp.2.markov_head.markov_w1.weight");
    if (!markov_w1 ||
        markov_w1->dtype != DS4_DFLASH_TENSOR_BF16 ||
        markov_w1->ndim != 2 ||
        markov_w1->shape[0] != cfg->vocab_size ||
        markov_w1->shape[1] != cfg->markov_rank) {
        return dflash_err(err, errlen, "DSpark Markov W1 tensor layout is invalid");
    }
    return ds4_dspark_tensor_read_bf16_f32(w,
                                           markov_w1,
                                           (uint64_t)prev_token_id * cfg->markov_rank,
                                           embedding,
                                           cfg->markov_rank,
                                           err,
                                           errlen);
}

int ds4_dspark_markov_logits_f32(const ds4_dspark_weights *w,
                                 const ds4_dspark_config *cfg,
                                 const float *prev_embedding,
                                 float *logits,
                                 char *err,
                                 size_t errlen) {
    const ds4_dflash_tensor *markov_w2 = NULL;
    if (!w || !w->loaded || !cfg || !cfg->loaded || !prev_embedding || !logits ||
        cfg->markov_rank == 0 || cfg->vocab_size == 0) {
        return dflash_err(err, errlen, "DSpark Markov logits request has invalid input");
    }
    markov_w2 = ds4_dspark_weights_find_tensor(w, "mtp.2.markov_head.markov_w2.weight");
    if (!markov_w2 ||
        markov_w2->dtype != DS4_DFLASH_TENSOR_BF16 ||
        markov_w2->ndim != 2 ||
        markov_w2->shape[0] != cfg->vocab_size ||
        markov_w2->shape[1] != cfg->markov_rank) {
        return dflash_err(err, errlen, "DSpark Markov W2 tensor layout is invalid");
    }
    return ds4_dspark_linear_bf16_f32(w,
                                      markov_w2,
                                      prev_embedding,
                                      logits,
                                      err,
                                      errlen);
}

int ds4_dspark_markov_logits_subset_f32(const ds4_dspark_weights *w,
                                        const ds4_dspark_config *cfg,
                                        const float *prev_embedding,
                                        const uint32_t *token_ids,
                                        uint32_t n_tokens,
                                        float *logits,
                                        char *err,
                                        size_t errlen) {
    const ds4_dflash_tensor *markov_w2 = NULL;
    const unsigned char *weight_data = NULL;
    const uint64_t rank = cfg ? cfg->markov_rank : 0;
    if (!w || !w->loaded || !cfg || !cfg->loaded || !prev_embedding ||
        !token_ids || !logits || rank == 0 || cfg->vocab_size == 0) {
        return dflash_err(err, errlen, "DSpark Markov subset request has invalid input");
    }
    markov_w2 = ds4_dspark_weights_find_tensor(w, "mtp.2.markov_head.markov_w2.weight");
    if (!markov_w2 ||
        markov_w2->dtype != DS4_DFLASH_TENSOR_BF16 ||
        markov_w2->ndim != 2 ||
        markov_w2->shape[0] != cfg->vocab_size ||
        markov_w2->shape[1] != cfg->markov_rank) {
        return dflash_err(err, errlen, "DSpark Markov W2 tensor layout is invalid");
    }
    if (dspark_tensor_data_ptr(w,
                               markov_w2,
                               0,
                               markov_w2->nbytes,
                               &weight_data,
                               err,
                               errlen) != 0) {
        return 1;
    }
    dspark_decode_tables_ensure();
    for (uint32_t i = 0; i < n_tokens; i++) {
        const uint32_t token_id = token_ids[i];
        if (token_id >= cfg->vocab_size) {
            return dflash_err(err, errlen, "DSpark Markov subset token id is out of range");
        }
        const unsigned char *row =
            weight_data + (uint64_t)token_id * rank * 2u;
        float acc = 0.0f;
        for (uint64_t col = 0; col < rank; col++) {
            const uint16_t raw = (uint16_t)row[col * 2u] |
                                 ((uint16_t)row[col * 2u + 1u] << 8);
            acc += prev_embedding[col] * dspark_bf16_table[raw];
        }
        logits[i] = acc;
    }
    return 0;
}

int ds4_dspark_confidence_logit_f32(const ds4_dspark_weights *w,
                                    const ds4_dspark_config *cfg,
                                    const float *hidden,
                                    const float *prev_embedding,
                                    float *logit,
                                    char *err,
                                    size_t errlen) {
    const ds4_dflash_tensor *proj = NULL;
    float *features = NULL;
    int rc = 1;
    if (!w || !w->loaded || !cfg || !cfg->loaded || !hidden || !prev_embedding ||
        !logit || cfg->hidden_size == 0 || cfg->markov_rank == 0 ||
        cfg->hidden_size > UINT32_MAX - cfg->markov_rank) {
        return dflash_err(err, errlen, "DSpark confidence logit request has invalid input");
    }
    proj = ds4_dspark_weights_find_tensor(w, "mtp.2.confidence_head.proj.weight");
    if (!proj ||
        proj->dtype != DS4_DFLASH_TENSOR_BF16 ||
        proj->ndim != 2 ||
        proj->shape[0] != 1 ||
        proj->shape[1] != (uint64_t)cfg->hidden_size + cfg->markov_rank) {
        return dflash_err(err, errlen, "DSpark confidence projection tensor layout is invalid");
    }
    features = calloc((size_t)proj->shape[1], sizeof(features[0]));
    if (!features) {
        return dflash_err(err, errlen, "out of memory running DSpark confidence head");
    }
    memcpy(features, hidden, (size_t)cfg->hidden_size * sizeof(features[0]));
    memcpy(features + cfg->hidden_size,
           prev_embedding,
           (size_t)cfg->markov_rank * sizeof(features[0]));
    rc = ds4_dspark_linear_bf16_f32(w,
                                    proj,
                                    features,
                                    logit,
                                    err,
                                    errlen);
    free(features);
    return rc;
}

static void dspark_corrected_logits_top2(const float *base_logits,
                                         const float *markov_logits,
                                         uint32_t vocab_size,
                                         uint32_t *top0,
                                         float *logit0,
                                         uint32_t *top1,
                                         float *logit1) {
    uint32_t b0 = 0;
    uint32_t b1 = 0;
    float v0 = -FLT_MAX;
    float v1 = -FLT_MAX;
    for (uint32_t i = 0; i < vocab_size; i++) {
        const float v = base_logits[i] + (markov_logits ? markov_logits[i] : 0.0f);
        if (v > v0) {
            b1 = b0;
            v1 = v0;
            b0 = i;
            v0 = v;
        } else if (v > v1) {
            b1 = i;
            v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

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
                                          size_t errlen) {
    const uint64_t vocab = cfg ? cfg->vocab_size : 0;
    const uint64_t hidden = cfg ? cfg->hidden_size : 0;
    const uint64_t markov_rank = cfg ? cfg->markov_rank : 0;
    float *prev_embedding = NULL;
    float *markov_logits = NULL;
    uint32_t prev_token = first_prev_token_id;
    uint32_t emitted = 0;
    const bool need_confidence =
        confidence_threshold > 0.0f || confidence_logits != NULL;
    const bool disable_markov = getenv("DS4_DSPARK_DISABLE_MARKOV") != NULL;
    int rc = -1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !base_logits || !draft_tokens ||
        proposal_len == 0 || vocab == 0 || hidden == 0 || markov_rank == 0 ||
        first_prev_token_id >= vocab) {
        dflash_err(err, errlen, "invalid DSpark draft token selection request");
        return -1;
    }
    if (proposal_len > cfg->block_size) {
        dflash_err(err, errlen, "DSpark proposal length exceeds configured block size");
        return -1;
    }
    if ((!disable_markov && vocab > SIZE_MAX / sizeof(markov_logits[0])) ||
        markov_rank > SIZE_MAX / sizeof(prev_embedding[0])) {
        dflash_err(err, errlen, "DSpark proposal selection buffers are too large");
        return -1;
    }
    if (need_confidence && !hidden_rows) {
        dflash_err(err, errlen, "DSpark confidence selection requires hidden rows");
        return -1;
    }

    if (!disable_markov || need_confidence) {
        prev_embedding = calloc((size_t)markov_rank, sizeof(prev_embedding[0]));
    }
    if (!disable_markov) {
        markov_logits = calloc((size_t)vocab, sizeof(markov_logits[0]));
    }
    if ((!disable_markov && !markov_logits) ||
        ((!disable_markov || need_confidence) && !prev_embedding)) {
        dflash_err(err, errlen, "out of memory selecting DSpark draft tokens");
        goto out;
    }

    for (uint32_t row = 0; row < proposal_len; row++) {
        uint32_t best = 0;
        uint32_t second = 0;
        float best_logit = 0.0f;
        float second_logit = 0.0f;
        float confidence_logit = 0.0f;

        if (!disable_markov || need_confidence) {
            if (ds4_dspark_markov_prev_embedding_f32(w,
                                                     cfg,
                                                     prev_token,
                                                     prev_embedding,
                                                     err,
                                                     errlen) != 0) {
                goto out;
            }
        }
        if (!disable_markov) {
            if (dspark_markov_logits_cached(w,
                                            cfg,
                                            prev_token,
                                            prev_embedding,
                                            markov_logits,
                                            err,
                                            errlen) != 0) {
                goto out;
            }
        }
        if (need_confidence &&
            ds4_dspark_confidence_logit_f32(w,
                                            cfg,
                                            hidden_rows + (uint64_t)row * hidden,
                                            prev_embedding,
                                            &confidence_logit,
                                            err,
                                            errlen) != 0) {
            goto out;
        }

        dspark_corrected_logits_top2(base_logits + (uint64_t)row * vocab,
                                     disable_markov ? NULL : markov_logits,
                                     cfg->vocab_size,
                                     &best,
                                     &best_logit,
                                     &second,
                                     &second_logit);
        (void)second;
        if (confidence_threshold > 0.0f &&
            dspark_sigmoidf(confidence_logit) < confidence_threshold) {
            break;
        }

        draft_tokens[emitted] = best;
        if (margins) margins[emitted] = best_logit - second_logit;
        if (confidence_logits) confidence_logits[emitted] = confidence_logit;
        emitted++;
        prev_token = best;
    }

    rc = (int)emitted;

out:
    free(markov_logits);
    free(prev_embedding);
    return rc;
}

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
                                               size_t errlen) {
    const uint64_t vocab = cfg ? cfg->vocab_size : 0;
    const uint64_t markov_rank = cfg ? cfg->markov_rank : 0;
    float *prev_embedding = NULL;
    float *markov_logits = NULL;
    uint32_t prev_token = first_prev_token_id;
    uint32_t emitted = 0;
    const bool disable_markov = getenv("DS4_DSPARK_DISABLE_MARKOV") != NULL;
    int rc = -1;

    if (!w || !w->loaded || !cfg || !cfg->loaded || !topk_token_ids ||
        !topk_base_logits || !draft_tokens || proposal_len == 0 ||
        top_k == 0 || vocab == 0 || markov_rank == 0 ||
        first_prev_token_id >= vocab) {
        dflash_err(err, errlen, "invalid DSpark top-k draft token selection request");
        return -1;
    }
    if (proposal_len > cfg->block_size ||
        (uint64_t)proposal_len > UINT64_MAX / top_k) {
        dflash_err(err, errlen, "DSpark top-k proposal shape is invalid");
        return -1;
    }
    if (markov_rank > SIZE_MAX / sizeof(prev_embedding[0])) {
        dflash_err(err, errlen, "DSpark top-k selection buffers are too large");
        return -1;
    }

    if (!disable_markov) {
        prev_embedding = calloc((size_t)markov_rank, sizeof(prev_embedding[0]));
        markov_logits = calloc((size_t)top_k, sizeof(markov_logits[0]));
        if (!prev_embedding || !markov_logits) {
            dflash_err(err, errlen, "out of memory selecting DSpark top-k draft tokens");
            goto out;
        }
    }

    for (uint32_t row = 0; row < proposal_len; row++) {
        const uint32_t *row_ids = topk_token_ids + (uint64_t)row * top_k;
        const float *row_logits = topk_base_logits + (uint64_t)row * top_k;
        uint32_t best = UINT32_MAX;
        uint32_t second = UINT32_MAX;
        float best_logit = -FLT_MAX;
        float second_logit = -FLT_MAX;

        if (!disable_markov) {
            if (ds4_dspark_markov_prev_embedding_f32(w,
                                                     cfg,
                                                     prev_token,
                                                     prev_embedding,
                                                     err,
                                                     errlen) != 0 ||
                ds4_dspark_markov_logits_subset_f32(w,
                                                    cfg,
                                                    prev_embedding,
                                                    row_ids,
                                                    top_k,
                                                    markov_logits,
                                                    err,
                                                    errlen) != 0) {
                goto out;
            }
        }

        for (uint32_t i = 0; i < top_k; i++) {
            const uint32_t token_id = row_ids[i];
            if (token_id >= vocab) continue;
            const float v = row_logits[i] +
                            (disable_markov ? 0.0f : markov_logits[i]);
            if (v > best_logit) {
                second = best;
                second_logit = best_logit;
                best = token_id;
                best_logit = v;
            } else if (v > second_logit) {
                second = token_id;
                second_logit = v;
            }
        }
        if (best == UINT32_MAX) break;
        (void)second;
        draft_tokens[emitted] = best;
        if (margins) margins[emitted] = best_logit - second_logit;
        emitted++;
        prev_token = best;
    }

    rc = (int)emitted;

out:
    free(markov_logits);
    free(prev_embedding);
    return rc;
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

static void rope_head_qwen3_inplace(float *x, uint32_t head_dim, uint32_t pos, float rope_theta) {
    const uint32_t half = head_dim / 2u;
    for (uint32_t i = 0; i < half; i++) {
        const float theta =
            powf(rope_theta, -((float)(2u * i) / (float)head_dim));
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

static bool dflash_attention_kv_visible(uint32_t q_noise_row,
                                        uint64_t kv_row,
                                        uint32_t n_target_rows,
                                        bool causal_noise_block) {
    uint64_t noise_kv_row = 0;

    if (!causal_noise_block || kv_row < n_target_rows) return true;
    noise_kv_row = kv_row - (uint64_t)n_target_rows;
    return noise_kv_row <= q_noise_row;
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
    const float rope_theta = cfg && cfg->rope_theta > 0.0f ?
        cfg->rope_theta : DS4_DFLASH_DEFAULT_ROPE_THETA;
    const bool causal_noise_block = cfg &&
        cfg->sliding_window > 0 &&
        !cfg->sliding_window_non_causal;
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
            rope_head_qwen3_inplace(head, cfg->head_dim, noise_positions[row], rope_theta);
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
            rope_head_qwen3_inplace(head, cfg->head_dim, target_positions[row], rope_theta);
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
            rope_head_qwen3_inplace(head, cfg->head_dim, noise_positions[row], rope_theta);
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
                if (!dflash_attention_kv_visible(row, kr, n_target_rows, causal_noise_block)) {
                    scores[kr] = -FLT_MAX;
                    continue;
                }
                scores[kr] = dot_f32_local(qh, kh, cfg->head_dim) * scale;
                if (scores[kr] > max_score) max_score = scores[kr];
            }
            memset(oh, 0, (size_t)cfg->head_dim * sizeof(oh[0]));
            float denom = 0.0f;
            for (uint64_t kr = 0; kr < total_kv_rows; kr++) {
                if (!dflash_attention_kv_visible(row, kr, n_target_rows, causal_noise_block)) {
                    continue;
                }
                const float *vh = v + kr * kv_dim + (uint64_t)kv_head * cfg->head_dim;
                const float weight = expf(scores[kr] - max_score);
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

static int select_token_row(const ds4_dflash_weights *w,
                            const ds4_dflash_config *cfg,
                            const ds4_dflash_tensor *d2t,
                            const ds4_dflash_tensor *t2d,
                            const float *row_logits,
                            uint32_t *draft_token,
                            uint32_t *target_token,
                            float *margin,
                            char *err,
                            size_t errlen) {
    uint32_t best = 0;
    float best_score = -FLT_MAX;
    float runner_score = -FLT_MAX;
    for (uint32_t tok = 0; tok < cfg->draft_vocab_size; tok++) {
        if (row_logits[tok] > best_score) {
            runner_score = best_score;
            best_score = row_logits[tok];
            best = tok;
        } else if (row_logits[tok] > runner_score) {
            runner_score = row_logits[tok];
        }
    }
    if (margin) *margin = runner_score > -FLT_MAX ? best_score - runner_score : 0.0f;
    /* Upstream stores d2t as an offset:
     * target_token_id = draft_token_id + d2t[draft_token_id]. */
    const int64_t offset = i64_data_at(w, d2t, best);
    const int64_t mapped = (int64_t)best + offset;
    if (mapped < 0 || (uint64_t)mapped >= cfg->vocab_size) {
        return dflash_err(err, errlen,
                          "DFlash draft token %u maps outside target vocab via offset %lld",
                          best,
                          (long long)offset);
    }
    if (!bool_data_at(w, t2d, (uint64_t)mapped)) {
        return dflash_err(err, errlen,
                          "DFlash draft token %u maps to inadmissible target token %lld",
                          best,
                          (long long)mapped);
    }
    *draft_token = best;
    *target_token = (uint32_t)mapped;
    return 0;
}

static int require_vocab_mapping_tensors(const ds4_dflash_weights *w,
                                         const ds4_dflash_config *cfg,
                                         const ds4_dflash_tensor **d2t,
                                         const ds4_dflash_tensor **t2d,
                                         char *err,
                                         size_t errlen) {
    if (!w || !w->loaded || !cfg || !cfg->loaded ||
        cfg->draft_vocab_size == 0 || cfg->vocab_size == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU token selection request");
    }
    if (require_tensor_1d(w, "d2t", DS4_DFLASH_TENSOR_I64, cfg->draft_vocab_size, d2t, err, errlen) != 0 ||
        require_tensor_1d(w, "t2d", DS4_DFLASH_TENSOR_BOOL, cfg->vocab_size, t2d, err, errlen) != 0) {
        return 1;
    }
    return 0;
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

    if (!logits || !draft_tokens || !target_tokens || n_rows == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU token selection request");
    }
    if (require_vocab_mapping_tensors(w, cfg, &d2t, &t2d, err, errlen) != 0) {
        return 1;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        if (select_token_row(w,
                             cfg,
                             d2t,
                             t2d,
                             logits + (uint64_t)row * cfg->draft_vocab_size,
                             &draft_tokens[row],
                             &target_tokens[row],
                             NULL,
                             err,
                             errlen) != 0) {
            return 1;
        }
    }

    return 0;
}

int ds4_dflash_cpu_select_draft_suffix_tokens(const ds4_dflash_weights *w,
                                              const ds4_dflash_config *cfg,
                                              const float *logits,
                                              uint32_t n_rows,
                                              uint32_t draft_cap,
                                              uint32_t *draft_tokens,
                                              uint32_t *target_tokens,
                                              float *margins,
                                              char *err,
                                              size_t errlen) {
    const ds4_dflash_tensor *d2t = NULL;
    const ds4_dflash_tensor *t2d = NULL;
    uint32_t row_base = 1u;
    const char *row_base_env = getenv("DS4_DFLASH_SELECT_ROW_BASE");

    if (!logits || !draft_tokens || !target_tokens || n_rows < 2 || draft_cap == 0) {
        return dflash_err(err, errlen, "invalid DFlash CPU draft suffix selection request");
    }
    if (draft_cap > n_rows - 1u) {
        return dflash_err(err, errlen, "DFlash draft suffix selection exceeds block rows");
    }
    if (row_base_env && row_base_env[0]) {
        char *end = NULL;
        unsigned long v = 0;
        errno = 0;
        v = strtoul(row_base_env, &end, 10);
        if (end == row_base_env || *end != '\0' || errno != 0 || v > UINT32_MAX) {
            return dflash_err(err, errlen, "invalid DS4_DFLASH_SELECT_ROW_BASE");
        }
        row_base = (uint32_t)v;
    }
    if (row_base > n_rows || draft_cap > n_rows - row_base) {
        return dflash_err(err, errlen, "DFlash draft suffix row base exceeds block rows");
    }
    if (require_vocab_mapping_tensors(w, cfg, &d2t, &t2d, err, errlen) != 0) {
        return 1;
    }

    for (uint32_t i = 0; i < draft_cap; i++) {
        const uint32_t row = row_base + i;
        if (select_token_row(w,
                             cfg,
                             d2t,
                             t2d,
                             logits + (uint64_t)row * cfg->draft_vocab_size,
                             &draft_tokens[i],
                             &target_tokens[i],
                             margins ? &margins[i] : NULL,
                             err,
                             errlen) != 0) {
            return 1;
        }
    }

    return 0;
}

int ds4_dflash_project_target_hidden(const ds4_dflash_weights *w,
                                     const ds4_dflash_config *cfg,
                                     const float *tap_hc,
                                     uint32_t n_tokens,
                                     uint32_t token_index,
                                     float *target_hidden,
                                     char *err,
                                     size_t errlen) {
    if (!w || !w->loaded || !cfg || !cfg->loaded || !tap_hc ||
        !target_hidden) {
        return dflash_err(err, errlen, "invalid DFlash target hidden projection request");
    }
    if (n_tokens == 0 || token_index >= n_tokens) {
        return dflash_err(err, errlen, "DFlash target hidden token index is out of range");
    }

    const ds4_dflash_tensor *fc = ds4_dflash_weights_find_tensor(w, "fc.weight");
    const ds4_dflash_tensor *hidden_norm = ds4_dflash_weights_find_tensor(w, "hidden_norm.weight");
    if (!fc || !hidden_norm) {
        return dflash_err(err, errlen, "DFlash target hidden projection tensors are not bound");
    }
    const uint64_t hidden = cfg->hidden_size;
    const uint64_t hc_dim = (uint64_t)cfg->hc_mult * hidden;
    const uint64_t fc_in = (uint64_t)cfg->n_target_layer_ids * hc_dim;
    if (fc->ndim != 2 || fc->shape[0] != hidden || fc->shape[1] != fc_in ||
        hidden_norm->ndim != 1 || hidden_norm->shape[0] != hidden) {
        return dflash_err(err, errlen, "DFlash target hidden projection tensor layout does not match config");
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

    return 0;
}

int ds4_dflash_prepare_noise_inputs(const ds4_dflash_weights *w,
                                    const ds4_dflash_config *cfg,
                                    uint32_t anchor_token,
                                    float *noise_embedding,
                                    char *err,
                                    size_t errlen) {
    if (!w || !w->loaded || !cfg || !cfg->loaded || !noise_embedding) {
        return dflash_err(err, errlen, "invalid DFlash noise input request");
    }
    if (anchor_token >= cfg->vocab_size || cfg->mask_token_id >= cfg->vocab_size) {
        return dflash_err(err, errlen, "DFlash noise input token id is outside target vocab");
    }

    const ds4_dflash_tensor *embed = ds4_dflash_weights_find_tensor(w, "embed_tokens.weight");
    if (!embed) {
        return dflash_err(err, errlen, "DFlash noise input embedding tensor is not bound");
    }
    const uint64_t hidden = cfg->hidden_size;
    if (embed->ndim != 2 || embed->shape[0] != cfg->vocab_size ||
        embed->shape[1] != hidden) {
        return dflash_err(err, errlen, "DFlash noise input embedding tensor layout does not match config");
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
    if (ds4_dflash_project_target_hidden(w,
                                         cfg,
                                         tap_hc,
                                         n_tokens,
                                         token_index,
                                         target_hidden,
                                         err,
                                         errlen) != 0 ||
        ds4_dflash_prepare_noise_inputs(w,
                                        cfg,
                                        anchor_token,
                                        noise_embedding,
                                        err,
                                        errlen) != 0) {
        return 1;
    }
    return 0;
}
