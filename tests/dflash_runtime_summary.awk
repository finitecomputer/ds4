/ds4: dflash spec drafted=/ {
    attempts++;
    line_has_rejections = 0;
    line_misses = 0;
    line_rejected = 0;
    for (i = 1; i <= NF; i++) {
        split($i, kv, "=");
        if (kv[1] == "drafted") drafted += kv[2] + 0;
        else if (kv[1] == "verified") verified += kv[2] + 0;
        else if (kv[1] == "accepted") accepted += kv[2] + 0;
        else if (kv[1] == "misses") {
            line_misses = kv[2] + 0;
            line_has_rejections = 1;
        } else if (kv[1] == "rejected_draft_tokens") {
            line_rejected = kv[2] + 0;
            line_has_rejections = 1;
        }
    }
    if (line_has_rejections) {
        summary_rejections_seen = 1;
        summary_misses += line_misses;
        summary_rejected += line_rejected;
    }
}

/ds4: dflash spec miss / {
    miss_at = -1;
    miss_drafted = 0;
    legacy_misses++;
    for (i = 1; i <= NF; i++) {
        split($i, kv, "=");
        if (kv[1] == "at") miss_at = kv[2] + 0;
        else if (kv[1] == "drafted") miss_drafted = kv[2] + 0;
    }
    if (miss_at >= 0 && miss_drafted > miss_at) {
        legacy_rejected += miss_drafted - miss_at;
    } else {
        legacy_rejected++;
    }
}

/ds4: dflash timing drafted=/ {
    timing++;
}

END {
    printf("attempts=%d\n", attempts);
    printf("drafted=%d\n", drafted);
    printf("verified=%d\n", verified);
    printf("accepted_including_anchor=%d\n", accepted);
    printf("misses=%d\n", summary_rejections_seen ? summary_misses : legacy_misses);
    printf("rejected_draft_tokens=%d\n", summary_rejections_seen ? summary_rejected : legacy_rejected);
    printf("timing_lines=%d\n", timing);
}
