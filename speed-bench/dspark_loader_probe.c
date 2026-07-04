#include "ds4_dflash.h"

#include <stdio.h>
#include <stdlib.h>

static double checksum(const float *x, uint64_t n) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += (double)x[i] * (double)((i % 17u) + 1u);
    }
    return sum;
}

static void print_routes(const ds4_dspark_stage_stats *stats) {
    printf("[");
    for (uint32_t i = 0; i < stats->n_routes; i++) {
        printf("%s%u:%.9f",
               i == 0 ? "" : " ",
               stats->route_indices[i],
               stats->route_weights[i]);
    }
    printf("]");
}

static uint32_t argmax_f32(const float *x, uint64_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

int main(int argc, char **argv) {
    char err[512] = {0};
    ds4_dspark_config cfg;
    ds4_dspark_weights weights;
    float *main_hidden = NULL;
    float *main_x = NULL;
    float *stage_a = NULL;
    float *stage_b = NULL;
    float *block_a = NULL;
    float *block_b = NULL;
    float *final_hidden = NULL;
    float *final_norm = NULL;
    float *block_final_hidden = NULL;
    float *block_final_norm = NULL;
    float *markov_embedding = NULL;
    float *markov_logits = NULL;
    uint64_t input_dim = 0;
    uint64_t hc_dim = 0;
    const uint64_t rope_position = 17;
    const uint32_t prev_token_id = 42;
    int rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/DeepSeek-V4-Flash-DSpark\n", argv[0]);
        return 2;
    }

    ds4_dspark_config_init(&cfg);
    ds4_dspark_weights_init(&weights);
    if (ds4_dspark_config_load(&cfg, argv[1], err, sizeof(err)) != 0 ||
        ds4_dspark_config_validate_target(&cfg, 4096, 129280, 43, err, sizeof(err)) != 0 ||
        ds4_dspark_weights_open_graph(&weights, argv[1], &cfg, err, sizeof(err)) != 0) {
        fprintf(stderr, "dspark probe: %s\n", err);
        goto out;
    }

    input_dim = (uint64_t)cfg.hidden_size * cfg.n_target_layer_ids;
    hc_dim = (uint64_t)cfg.hidden_size * cfg.hc_mult;
    main_hidden = calloc((size_t)input_dim, sizeof(main_hidden[0]));
    main_x = calloc(cfg.hidden_size, sizeof(main_x[0]));
    stage_a = calloc((size_t)hc_dim, sizeof(stage_a[0]));
    stage_b = calloc((size_t)hc_dim, sizeof(stage_b[0]));
    block_a = calloc((size_t)cfg.block_size * hc_dim, sizeof(block_a[0]));
    block_b = calloc((size_t)cfg.block_size * hc_dim, sizeof(block_b[0]));
    final_hidden = calloc(cfg.hidden_size, sizeof(final_hidden[0]));
    final_norm = calloc(cfg.hidden_size, sizeof(final_norm[0]));
    block_final_hidden = calloc((size_t)cfg.block_size * cfg.hidden_size,
                                sizeof(block_final_hidden[0]));
    block_final_norm = calloc((size_t)cfg.block_size * cfg.hidden_size,
                              sizeof(block_final_norm[0]));
    markov_embedding = calloc(cfg.markov_rank, sizeof(markov_embedding[0]));
    markov_logits = calloc(cfg.vocab_size, sizeof(markov_logits[0]));
    if (!main_hidden || !main_x || !stage_a || !stage_b || !block_a || !block_b ||
        !final_hidden || !final_norm || !block_final_hidden || !block_final_norm ||
        !markov_embedding || !markov_logits) {
        fprintf(stderr, "dspark probe: out of memory\n");
        goto out;
    }
    for (uint64_t i = 0; i < input_dim; i++) {
        main_hidden[i] = (float)((int)(i % 23u) - 11) * 0.03125f;
    }

    if (ds4_dspark_project_main_hidden(&weights,
                                       &cfg,
                                       main_hidden,
                                       main_x,
                                       err,
                                       sizeof(err)) != 0) {
        fprintf(stderr, "dspark probe: %s\n", err);
        goto out;
    }
    if (ds4_dspark_init_hc_block_from_main_f32(&cfg,
                                               main_x,
                                               1,
                                               stage_a,
                                               err,
                                               sizeof(err)) != 0 ||
        ds4_dspark_init_hc_block_from_main_f32(&cfg,
                                               main_x,
                                               cfg.block_size,
                                               block_a,
                                               err,
                                               sizeof(err)) != 0) {
        fprintf(stderr, "dspark probe: %s\n", err);
        goto out;
    }
    for (uint32_t row = 0; row < cfg.block_size; row++) {
        const float jitter = (float)((int)(row % 5u) - 2) * 0.001953125f;
        for (uint64_t i = 0; i < hc_dim; i++) {
            block_a[(uint64_t)row * hc_dim + i] += jitter;
        }
    }

    printf("dspark probe: loaded %u shards, bound %u tensors\n",
           weights.n_shards,
           weights.n_bound_tensors);
    printf("dspark probe: block=%u hidden=%u target_layers=%u markov_rank=%u sliding_window=%u stages=%u\n",
           cfg.block_size,
           cfg.hidden_size,
           cfg.n_target_layer_ids,
           cfg.markov_rank,
           cfg.sliding_window,
           cfg.n_mtp_stages);
    printf("dspark probe: main_x checksum=%.9f first=[%.9f %.9f %.9f %.9f]\n",
           checksum(main_x, cfg.hidden_size),
           main_x[0],
           main_x[1],
           main_x[2],
           main_x[3]);

    if (getenv("DS4_DSPARK_PROBE_SKIP_SINGLE") == NULL) {
        for (uint32_t stage = 0; stage < cfg.n_mtp_stages; stage++) {
            ds4_dspark_stage_stats stats;
            ds4_dspark_stage_stats_init(&stats);
            if (ds4_dspark_run_stage_reference(&weights,
                                               &cfg,
                                               stage,
                                               rope_position + stage,
                                               stage_a,
                                               stage_b,
                                               &stats,
                                               err,
                                               sizeof(err)) != 0) {
                fprintf(stderr, "dspark probe: %s\n", err);
                goto out;
            }
            printf("dspark probe: mtp%u.stage rope_pos=%llu attn_hc_checksum=%.9f ffn_hc_checksum=%.9f ffn_out_checksum=%.9f routed_sum_checksum=%.9f shared_out_checksum=%.9f routes=",
                   stage,
                   (unsigned long long)(rope_position + stage),
                   stats.attn_hc_checksum,
                   stats.ffn_hc_checksum,
                   stats.ffn_out_checksum,
                   stats.routed_sum_checksum,
                   stats.shared_out_checksum);
            print_routes(&stats);
            printf(" first=[%.9f %.9f %.9f %.9f]\n",
                   stats.ffn_hc_first[0],
                   stats.ffn_hc_first[1],
                   stats.ffn_hc_first[2],
                   stats.ffn_hc_first[3]);

            float *tmp = stage_a;
            stage_a = stage_b;
            stage_b = tmp;
        }
    } else {
        printf("dspark probe: single-row stage diagnostics skipped by DS4_DSPARK_PROBE_SKIP_SINGLE=1\n");
    }

    for (uint32_t stage = 0; stage < cfg.n_mtp_stages; stage++) {
        if (ds4_dspark_run_stage_block_reference(&weights,
                                                 &cfg,
                                                 stage,
                                                 rope_position,
                                                 block_a,
                                                 cfg.block_size,
                                                 block_b,
                                                 err,
                                                 sizeof(err)) != 0) {
            fprintf(stderr, "dspark probe: %s\n", err);
            goto out;
        }
        printf("dspark probe: mtp%u.block rows=%u checksum=%.9f row0=[%.9f %.9f %.9f %.9f] row%u=[%.9f %.9f %.9f %.9f]\n",
               stage,
               cfg.block_size,
               checksum(block_b, cfg.block_size * hc_dim),
               block_b[0],
               block_b[1],
               block_b[2],
               block_b[3],
               cfg.block_size - 1u,
               block_b[((uint64_t)(cfg.block_size - 1u) * cfg.hc_mult) * cfg.hidden_size],
               block_b[((uint64_t)(cfg.block_size - 1u) * cfg.hc_mult) * cfg.hidden_size + 1u],
               block_b[((uint64_t)(cfg.block_size - 1u) * cfg.hc_mult) * cfg.hidden_size + 2u],
               block_b[((uint64_t)(cfg.block_size - 1u) * cfg.hc_mult) * cfg.hidden_size + 3u]);

        float *tmp = block_a;
        block_a = block_b;
        block_b = tmp;
    }

    if (ds4_dspark_final_hc_head_f32(&weights,
                                     &cfg,
                                     stage_a,
                                     final_hidden,
                                     err,
                                     sizeof(err)) != 0 ||
        ds4_dspark_final_norm_f32(&weights,
                                  &cfg,
                                  final_hidden,
                                  final_norm,
                                  err,
                                  sizeof(err)) != 0) {
        fprintf(stderr, "dspark probe: %s\n", err);
        goto out;
    }
    printf("dspark probe: final_hc checksum=%.9f norm_checksum=%.9f first=[%.9f %.9f %.9f %.9f]\n",
           checksum(final_hidden, cfg.hidden_size),
           checksum(final_norm, cfg.hidden_size),
           final_norm[0],
           final_norm[1],
           final_norm[2],
           final_norm[3]);

    if (getenv("DS4_DSPARK_PROBE_SKIP_MARKOV") != NULL) {
        printf("dspark probe: Markov diagnostics skipped by DS4_DSPARK_PROBE_SKIP_MARKOV=1\n");
    } else if (ds4_dspark_markov_prev_embedding_f32(&weights,
                                                    &cfg,
                                                    prev_token_id,
                                                    markov_embedding,
                                                    err,
                                                    sizeof(err)) != 0 ||
               ds4_dspark_markov_logits_f32(&weights,
                                            &cfg,
                                            markov_embedding,
                                            markov_logits,
                                            err,
                                            sizeof(err)) != 0) {
        fprintf(stderr, "dspark probe: %s\n", err);
        goto out;
    } else {
        const uint32_t markov_top = argmax_f32(markov_logits, cfg.vocab_size);
        float confidence = 0.0f;
        if (ds4_dspark_confidence_logit_f32(&weights,
                                            &cfg,
                                            final_norm,
                                            markov_embedding,
                                            &confidence,
                                            err,
                                            sizeof(err)) != 0) {
            fprintf(stderr, "dspark probe: %s\n", err);
            goto out;
        }
        printf("dspark probe: markov prev_token=%u embedding_checksum=%.9f logits_checksum=%.9f top=%u top_logit=%.9f confidence_logit=%.9f\n",
               prev_token_id,
               checksum(markov_embedding, cfg.markov_rank),
               checksum(markov_logits, cfg.vocab_size),
               markov_top,
               markov_logits[markov_top],
               confidence);
    }
    if (ds4_dspark_final_block_norm_f32(&weights,
                                        &cfg,
                                        block_a,
                                        cfg.block_size,
                                        block_final_hidden,
                                        block_final_norm,
                                        err,
                                        sizeof(err)) != 0) {
        fprintf(stderr, "dspark probe: %s\n", err);
        goto out;
    }
    printf("dspark probe: block_final rows=%u hidden_checksum=%.9f norm_checksum=%.9f row0=[%.9f %.9f %.9f %.9f] row%u=[%.9f %.9f %.9f %.9f]\n",
           cfg.block_size,
           checksum(block_final_hidden, cfg.block_size * cfg.hidden_size),
           checksum(block_final_norm, cfg.block_size * cfg.hidden_size),
           block_final_norm[0],
           block_final_norm[1],
           block_final_norm[2],
           block_final_norm[3],
           cfg.block_size - 1u,
           block_final_norm[(uint64_t)(cfg.block_size - 1u) * cfg.hidden_size],
           block_final_norm[(uint64_t)(cfg.block_size - 1u) * cfg.hidden_size + 1u],
           block_final_norm[(uint64_t)(cfg.block_size - 1u) * cfg.hidden_size + 2u],
           block_final_norm[(uint64_t)(cfg.block_size - 1u) * cfg.hidden_size + 3u]);

    rc = 0;

out:
    free(main_hidden);
    free(main_x);
    free(stage_a);
    free(stage_b);
    free(block_a);
    free(block_b);
    free(final_hidden);
    free(final_norm);
    free(block_final_hidden);
    free(block_final_norm);
    free(markov_embedding);
    free(markov_logits);
    ds4_dspark_weights_free(&weights);
    ds4_dspark_config_free(&cfg);
    return rc;
}
