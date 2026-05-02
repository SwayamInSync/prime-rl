"""Orchestrator-side rollout filters for detecting degenerate generations.

Filters run after rollouts complete, inspecting token IDs and logprobs to
detect gibberish or repetition. Detection metrics are always tracked.
When enforce=True, detected rollouts are skipped entirely during training and
are not sent to the trainer. Reward is kept as-is for baseline calculation.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import verifiers as vf

from prime_rl.configs.orchestrator import FilterConfig
from prime_rl.utils.logger import get_logger


@dataclass
class FilterResult:
    detected: bool
    detection_index: int | None = None


class RolloutFilter(Protocol):
    name: str
    enforce: bool

    def check(self, rollout: vf.RolloutOutput) -> FilterResult: ...


@dataclass
class GibberishFilter:
    """Flags rollouts containing rare tokens generated at high entropy.

    A token is flagged when both:
      - id(token) > token_id_threshold  (rare BPE token)
      - logprob(token) < -log(vocab_size) - logprob_offset  (high entropy)

    References:
      Section 5.2, https://arxiv.org/abs/2510.02387
    """

    name: str
    token_id_threshold: int
    logprob_threshold: float
    enforce: bool = False

    def check(self, rollout: vf.RolloutOutput) -> FilterResult:
        global_idx = 0
        for step in rollout["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for token_id, logprob in zip(tokens["completion_ids"], tokens["completion_logprobs"]):
                if token_id > self.token_id_threshold and logprob < self.logprob_threshold:
                    return FilterResult(detected=True, detection_index=global_idx)
                global_idx += 1
        return FilterResult(detected=False)


@dataclass
class RepetitionFilter:
    """Flags rollouts with pathological repetition loops.

    Counts consecutive tokens where logprob > log(prob_threshold), indicating
    the model is generating with very high confidence. When the streak reaches
    the window size, the rollout is flagged.

    References:
      Section 3.2, https://arxiv.org/abs/2506.13585
    """

    name: str
    window: int
    logprob_threshold: float
    enforce: bool = False

    def check(self, rollout: vf.RolloutOutput) -> FilterResult:
        consecutive = 0
        global_idx = 0
        for step in rollout["trajectory"]:
            tokens = step["tokens"]
            if tokens is None:
                continue
            for logprob in tokens["completion_logprobs"]:
                if logprob > self.logprob_threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= self.window:
                    return FilterResult(detected=True, detection_index=global_idx)
                global_idx += 1
        return FilterResult(detected=False)


@dataclass
class ZeroAdvantageFilter:
    """Flags rollouts with zero advantage.

    This filter is applied after advantages are computed and checks if the
    rollout's advantage field is zero.
    """

    name: str
    enforce: bool = True

    def check(self, rollout: vf.RolloutOutput) -> FilterResult:
        advantage = rollout.get("advantage")
        if advantage is not None and advantage == 0.0:
            return FilterResult(detected=True)
        return FilterResult(detected=False)


def setup_filter(config: FilterConfig, vocab_size: int) -> RolloutFilter:
    """Create a RolloutFilter from a filter config."""
    if config.type == "gibberish":
        return GibberishFilter(
            name="gibberish",
            token_id_threshold=config.token_id_threshold,
            logprob_threshold=-math.log(vocab_size) - config.logprob_offset,
            enforce=config.enforce,
        )
    elif config.type == "repetition":
        return RepetitionFilter(
            name="repetition",
            window=config.window,
            logprob_threshold=math.log(config.prob_threshold),
            enforce=config.enforce,
        )
    elif config.type == "zero_advantage":
        return ZeroAdvantageFilter(
            name="zero_advantage",
            enforce=config.enforce,
        )
    raise ValueError(f"Unknown filter type: {config.type}")


def setup_filters(configs: list[FilterConfig], vocab_size: int) -> list[RolloutFilter]:
    """Create RolloutFilters from a list of filter configs."""
    filters = [setup_filter(config, vocab_size) for config in configs]
    if filters:
        get_logger().info(f"Configured {len(filters)} rollout filter(s):")
        for config, filt in zip(configs, filters):
            mode = "Enforcing" if filt.enforce else "Monitoring"
            params = ", ".join(f"{k}={v}" for k, v in config.model_dump().items())
            get_logger().info(f"  {mode} {filt.name} filter ({params})")
    return filters


def apply_filters(filters: list[RolloutFilter], rollouts: list[vf.RolloutOutput]) -> None:
    """Flag rollouts in-place with per-filter detection and drop decision.

    Each rollout gets a `filters` dict with per-filter detection booleans and
    an `is_filtered` bool that is True iff an enforcing filter detected it.
    First matching filter wins per rollout (no double-counting). Reward and
    trajectory tokens are left untouched so the rollout can still contribute
    to baseline calculations and metric aggregation.
    """
    for rollout in rollouts:
        rollout["filters"] = {f.name: False for f in filters}
        rollout["is_filtered"] = False

    if not filters:
        return

    counts: dict[str, int] = {f.name: 0 for f in filters}
    total_detected = 0
    total_enforced = 0

    for rollout in rollouts:
        for filt in filters:
            result = filt.check(rollout)
            if result.detected:
                counts[filt.name] += 1
                total_detected += 1
                rollout["filters"][filt.name] = True
                if filt.enforce:
                    rollout["is_filtered"] = True
                    total_enforced += 1
                break

    if total_detected > 0:
        enforced_msg = f", enforced {total_enforced}" if total_enforced > 0 else ""
        get_logger().info(
            f"Detected {total_detected}/{len(rollouts)} rollouts "
            f"({', '.join(f'{name}={c}' for name, c in counts.items() if c > 0)})" + enforced_msg
        )


def select_useful_groups(
    pool: list["vf.RolloutOutput"],
    rollouts_per_example: int,
    target_groups: int,
    pad_with_filtered: bool,
) -> tuple[list["vf.RolloutOutput"], list[bool], int]:
    """DAPO-style group selection.

    Pre-condition: ``apply_filters`` has been called on every rollout in ``pool``,
    and ``len(pool) % rollouts_per_example == 0`` (groups stay intact).

    Selection rule (deterministic, no randomness, no reward-based reordering):
      1. Mark each group as *useful* iff at least one of its rollouts has
         ``is_filtered == False``.
      2. Take useful groups in arrival order until ``target_groups`` are
         collected (or the supply is exhausted).
      3. If still short of ``target_groups`` and ``pad_with_filtered`` is True,
         pad from the head of the filtered groups (also in arrival order).
      4. Restore arrival order in the shipped batch (avoids re-ordering bias
         relative to other downstream metric aggregation).

    Returns:
      shipped: flat rollout list (length is a multiple of rollouts_per_example).
      useful_mask: per-group usefulness flag for the *full* pool (len == n_groups).
      padded_groups: number of filtered (zero-advantage) groups added as padding.
    """
    K = rollouts_per_example
    if K <= 0:
        raise ValueError(f"rollouts_per_example must be > 0, got {K}")
    if len(pool) % K != 0:
        raise ValueError(
            f"pool size {len(pool)} is not a multiple of rollouts_per_example {K}; "
            "groups must remain intact across the dynamic-sampling loop"
        )
    n_groups = len(pool) // K
    useful_mask = [
        any(not pool[g * K + i].get("is_filtered", False) for i in range(K))
        for g in range(n_groups)
    ]
    useful_indices = [g for g in range(n_groups) if useful_mask[g]]
    filtered_indices = [g for g in range(n_groups) if not useful_mask[g]]

    chosen = useful_indices[:target_groups]
    padded_groups = 0
    if len(chosen) < target_groups and pad_with_filtered:
        pad = filtered_indices[: target_groups - len(chosen)]
        padded_groups = len(pad)
        chosen.extend(pad)
    chosen.sort()  # preserve arrival order in shipped batch

    shipped: list = []
    for g in chosen:
        shipped.extend(pool[g * K : (g + 1) * K])
    return shipped, useful_mask, padded_groups
