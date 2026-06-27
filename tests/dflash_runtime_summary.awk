/ds4: dflash spec drafted=/ {
    attempts++;
    for (i = 1; i <= NF; i++) {
        split($i, kv, "=");
        if (kv[1] == "drafted") drafted += kv[2] + 0;
        else if (kv[1] == "verified") verified += kv[2] + 0;
        else if (kv[1] == "accepted") accepted += kv[2] + 0;
    }
}

/ds4: dflash spec miss / {
    miss_at = -1;
    miss_drafted = 0;
    misses++;
    for (i = 1; i <= NF; i++) {
        split($i, kv, "=");
        if (kv[1] == "at") miss_at = kv[2] + 0;
        else if (kv[1] == "drafted") miss_drafted = kv[2] + 0;
    }
    if (miss_at >= 0 && miss_drafted > miss_at) {
        rejected += miss_drafted - miss_at;
    } else {
        rejected++;
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
    printf("misses=%d\n", misses);
    printf("rejected_draft_tokens=%d\n", rejected);
    printf("timing_lines=%d\n", timing);
}
