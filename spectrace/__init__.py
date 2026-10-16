"""SpecTrace: score speculative-decoding drafters and check inference engines without re-running the target model."""
from .greedy import passes_from_ranks, tree_tokens
from .sampling import expected_passes_from_log
from .ngram import vllm_ngram_propose, replay_deterministic

__all__ = ["passes_from_ranks", "tree_tokens", "expected_passes_from_log", "vllm_ngram_propose", "replay_deterministic"]
__version__ = "0.1.0"
