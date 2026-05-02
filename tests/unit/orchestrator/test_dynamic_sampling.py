"""Tests for DAPO-style dynamic-sampling group selection (`select_useful_groups`).

These tests exercise the deterministic, in-memory selection logic without
spinning up the full orchestrator. Invariants under test:

  * Shipped batch size is exactly ``target_groups * rollouts_per_example``
    when supply is sufficient (or padding is enabled).
  * Groups stay intact (we never split a group across the shipped/dropped
    boundary).
  * Selection is by arrival order (no reward-based reordering → no bias).
  * Padding with filtered groups, when enabled, fills the remainder; padded
    groups all have ``is_filtered=True``.
  * Per-group "useful" flag matches the per-rollout ``is_filtered`` ground
    truth.
"""

from __future__ import annotations

import pytest

from prime_rl.orchestrator.filters import select_useful_groups


def _grp(filtered_per_rollout: list[bool], gid: int) -> list[dict]:
    return [
        {"is_filtered": f, "_gid": gid, "_ridx": i, "advantage": (0.0 if f else 1.0)}
        for i, f in enumerate(filtered_per_rollout)
    ]


def _pool(groups: list[list[bool]]) -> list[dict]:
    out: list[dict] = []
    for gid, g in enumerate(groups):
        out.extend(_grp(g, gid))
    return out


def _gids(rollouts: list[dict]) -> list[int]:
    return sorted({r["_gid"] for r in rollouts})


def test_basic_supply_meets_target():
    K = 4
    pool = _pool(
        [
            [False, True, False, True],   # group 0 useful
            [True, True, True, True],     # group 1 NOT useful (all filtered)
            [False, False, False, False], # group 2 useful
            [False, True, True, True],    # group 3 useful
        ]
    )
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=2, pad_with_filtered=True)
    assert mask == [True, False, True, True]
    assert padded == 0
    assert len(shipped) == 2 * K
    # First two useful groups are 0 and 2 — must be selected, in original order.
    assert _gids(shipped) == [0, 2]


def test_take_useful_in_arrival_order_not_by_advantage():
    K = 2
    # Three useful groups, target_groups=2 → must take groups 0 and 2 (the first
    # two useful), NOT, e.g., the highest-reward ones. This guards against
    # reward-based selection bias.
    pool = _pool(
        [
            [False, True],  # 0 useful
            [True, True],   # 1 NOT useful
            [False, True],  # 2 useful
            [False, True],  # 3 useful (must be DROPPED — arrived later)
        ]
    )
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=2, pad_with_filtered=True)
    assert mask == [True, False, True, True]
    assert padded == 0
    assert _gids(shipped) == [0, 2]


def test_pad_with_filtered_when_supply_short():
    K = 2
    pool = _pool(
        [
            [False, True],   # 0 useful
            [True, True],    # 1 NOT useful
            [True, True],    # 2 NOT useful
        ]
    )
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=3, pad_with_filtered=True)
    assert mask == [True, False, False]
    assert padded == 2
    # All 3 groups shipped (0 useful, plus padding from 1 and 2 in arrival order).
    assert _gids(shipped) == [0, 1, 2]
    assert len(shipped) == 3 * K


def test_pad_disabled_returns_short_batch():
    K = 2
    pool = _pool(
        [
            [False, True],   # 0 useful
            [True, True],    # 1 NOT useful
            [True, True],    # 2 NOT useful
        ]
    )
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=3, pad_with_filtered=False)
    assert padded == 0
    # Only the useful group is shipped; trainer would see a smaller batch.
    assert _gids(shipped) == [0]
    assert len(shipped) == 1 * K


def test_arrival_order_preserved_in_shipped_batch():
    K = 3
    pool = _pool(
        [
            [True, True, True],     # 0 NOT useful
            [False, True, True],    # 1 useful
            [True, True, True],     # 2 NOT useful
            [False, False, True],   # 3 useful
        ]
    )
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=2, pad_with_filtered=True)
    assert padded == 0
    # Selected useful groups: 1 and 3 in arrival order.
    assert _gids(shipped) == [1, 3]
    # Within each group, per-rollout order must also be preserved.
    g1 = [r for r in shipped if r["_gid"] == 1]
    assert [r["_ridx"] for r in g1] == [0, 1, 2]


def test_groups_stay_intact_no_partial_groups():
    K = 4
    pool = _pool(
        [
            [False, True, False, True],
            [True, True, True, True],
            [False, False, False, False],
        ]
    )
    shipped, _, _ = select_useful_groups(pool, K, target_groups=2, pad_with_filtered=True)
    # Every shipped group must contain exactly K rollouts.
    by_gid: dict[int, int] = {}
    for r in shipped:
        by_gid[r["_gid"]] = by_gid.get(r["_gid"], 0) + 1
    for gid, count in by_gid.items():
        assert count == K, f"group {gid} arrived with {count} rollouts, expected {K}"


def test_pool_size_must_be_multiple_of_k():
    K = 3
    bad_pool = [{"is_filtered": False} for _ in range(7)]  # 7 not divisible by 3
    with pytest.raises(ValueError, match="not a multiple"):
        select_useful_groups(bad_pool, K, target_groups=1, pad_with_filtered=True)


def test_target_zero_returns_empty():
    K = 2
    pool = _pool([[False, False], [False, True]])
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=0, pad_with_filtered=True)
    assert shipped == []
    assert padded == 0
    assert mask == [True, True]


def test_all_groups_useful_takes_first_n():
    K = 2
    pool = _pool([[False, True]] * 5)
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=3, pad_with_filtered=True)
    assert mask == [True] * 5
    assert padded == 0
    assert _gids(shipped) == [0, 1, 2]


def test_all_groups_filtered_pad_fills_completely():
    K = 2
    pool = _pool([[True, True]] * 4)
    shipped, mask, padded = select_useful_groups(pool, K, target_groups=2, pad_with_filtered=True)
    assert mask == [False] * 4
    assert padded == 2
    assert _gids(shipped) == [0, 1]


def test_useful_mask_has_one_entry_per_group():
    K = 4
    pool = _pool([[False] * 4, [True] * 4, [False, True, False, True]])
    _, mask, _ = select_useful_groups(pool, K, target_groups=1, pad_with_filtered=True)
    assert len(mask) == 3
