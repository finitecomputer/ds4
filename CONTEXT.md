# DS4 Glossary

## DS4 / DwarfStar

A narrow native inference engine for DeepSeek V4-family local inference. DS4 is
not a generic GGUF runner.

## Raw Sliding-Window KV Cache

The high-resolution attention KV cache for the most recent local context tokens.
It is separate from the compressed attention history.

## Compressed Attention KV Cache

The older attention history stored as compressed rows rather than one full KV row
per token. This is the target cache for the CUDA F16 compressed-KV experiment.

## Indexer Compressed Cache

The separate compressed stream used by selective compressed-attention layers to
choose which compressed attention rows are visible. It is not part of the first
CUDA F16 compressed-KV implementation.

## CUDA F16 Compressed Attention KV

An opt-in CUDA storage mode where the compressed attention KV cache is stored as
F16 while the existing CUDA default remains unchanged until validation supports
changing it. The first implementation scope excludes raw sliding-window KV and
the indexer compressed cache. The initial opt-in surface is compile-time only,
not a CLI or server runtime option.
