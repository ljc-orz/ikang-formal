#!/usr/bin/env python3
"""生成 50,000 行的训练集、内部验证集和外部验证集。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pymysql
from pymysql.cursors import DictCursor

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "datas",
    "charset": "utf8mb4",
    "unix_socket": "/DaTa/mysql/mysql.sock",
    "local_infile": True,
    "autocommit": False,
}

SOURCE_TABLE = "result_merged_wide_filtered"
OUTPUT_TABLE = "result_merged_wide_split_50000"
TARGET_DATASET_ROWS = 50_000
EXCLUDED_IMAGE_PREFIX = "Airdoc-Data3"

INDICATORS = [
    "result_alt",
    "result_bmi",
    "result_fbg",
    "result_hb",
    "result_hba1c",
    "result_hct",
    "result_hdl_c",
    "result_rbc",
    "result_scr",
    "result_tg",
    "result_wbc",
]

TRAIN = "train"
INTERNAL = "internal_validation"
EXTERNAL = "external_validation"

RANDOM_SEED = 20260909
N_RESTARTS = 200
MAX_SWAP_STEPS = 100
RATE_WEIGHT = 1.0
ROW_BALANCE_WEIGHT = 0.50
OBSERVED_BALANCE_WEIGHT = 0.25
FEASIBILITY_PENALTY = 50.0
WORST_RATE_GAP_WEIGHT = 0.75
EXACT_ENUMERATION_MAX_HOSPITALS = 16
JOINT_HEURISTIC_RESTARTS = 20
JOINT_MAX_SWAP_STEPS = 100
EPS = 1e-12

SPLIT_NAMES = np.asarray([TRAIN, INTERNAL, EXTERNAL], dtype=object)
SPLIT_TARGETS = np.asarray([0.60, 0.20, 0.20], dtype=np.float64)


@dataclass
class ZoneData:
    zone: str
    hospids: np.ndarray
    rows: np.ndarray
    observed: np.ndarray
    positive: np.ndarray


def fetch_eligible_rows(conn: pymysql.connections.Connection) -> list[dict[str, Any]]:
    """过滤 Airdoc-Data3，并读取抽样和分层所需字段。"""
    metric_sql = ", ".join(f"`{name}`" for name in INDICATORS)
    sql = f"""
        SELECT uid, id, economic_zone, hospid, {metric_sql}
        FROM `{SOURCE_TABLE}`
        WHERE economic_zone IS NOT NULL
          AND hospid IS NOT NULL
          AND (image_path_1 IS NULL OR image_path_1 NOT LIKE %s)
          AND (image_path_2 IS NULL OR image_path_2 NOT LIKE %s)
        ORDER BY economic_zone, hospid, uid, id
    """
    pattern = EXCLUDED_IMAGE_PREFIX + "%"
    with conn.cursor() as cursor:
        cursor.execute(sql, (pattern, pattern))
        return list(cursor.fetchall())


def proportional_sample_by_hospital(
    rows: list[dict[str, Any]], target: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """按医院行数等比例抽样，并严格得到 target 行。"""
    if target <= 0:
        raise ValueError("TARGET_DATASET_ROWS 必须大于 0")
    if len(rows) < target:
        raise ValueError(
            f"过滤后且地区、医院非空的数据只有 {len(rows)} 行，少于目标 {target} 行"
        )

    groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["economic_zone"]), int(row["hospid"]))
        groups.setdefault(key, []).append(index)

    keys = list(groups)
    sizes = np.asarray([len(groups[key]) for key in keys], dtype=np.int64)
    ideal = sizes.astype(np.float64) * target / len(rows)
    quotas = np.floor(ideal).astype(np.int64)
    if target >= len(keys):
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, sizes)

    while int(quotas.sum()) < target:
        candidates = np.flatnonzero(quotas < sizes)
        order = candidates[np.argsort(-(ideal[candidates] - quotas[candidates]))]
        take = min(target - int(quotas.sum()), len(order))
        quotas[order[:take]] += 1

    while int(quotas.sum()) > target:
        lower_bound = 1 if target >= len(keys) else 0
        candidates = np.flatnonzero(quotas > lower_bound)
        order = candidates[np.argsort(ideal[candidates] - quotas[candidates])]
        take = min(int(quotas.sum()) - target, len(order))
        quotas[order[:take]] -= 1

    sampled_indices: list[int] = []
    for key, quota in zip(keys, quotas):
        indices = np.asarray(groups[key], dtype=np.int64)
        chosen = rng.choice(indices, size=int(quota), replace=False)
        sampled_indices.extend(int(index) for index in chosen)
    rng.shuffle(sampled_indices)
    assert len(sampled_indices) == target
    return [rows[index] for index in sampled_indices]


def build_zone_data(rows: list[dict[str, Any]]) -> list[ZoneData]:
    """把抽中的行聚合成地区—医院级统计量。"""
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        zone = str(row["economic_zone"])
        hospid = int(row["hospid"])
        hospital = grouped.setdefault(zone, {}).setdefault(
            hospid,
            {
                "row_count": 0,
                "observed": np.zeros(len(INDICATORS), dtype=np.float64),
                "positive": np.zeros(len(INDICATORS), dtype=np.float64),
            },
        )
        hospital["row_count"] += 1
        for j, indicator in enumerate(INDICATORS):
            value = row[indicator]
            if value in (0, 1):
                hospital["observed"][j] += 1
                hospital["positive"][j] += int(value == 1)

    result = []
    for zone in sorted(grouped):
        hospitals = grouped[zone]
        hospids = sorted(hospitals)
        result.append(
            ZoneData(
                zone=zone,
                hospids=np.asarray(hospids, dtype=np.int64),
                rows=np.asarray(
                    [hospitals[h]["row_count"] for h in hospids], dtype=np.float64
                ),
                observed=np.asarray(
                    [hospitals[h]["observed"] for h in hospids], dtype=np.float64
                ),
                positive=np.asarray(
                    [hospitals[h]["positive"] for h in hospids], dtype=np.float64
                ),
            )
        )
    return result


def subset_zone(data: ZoneData, indices: np.ndarray) -> ZoneData:
    return ZoneData(
        data.zone,
        data.hospids[indices],
        data.rows[indices],
        data.observed[indices],
        data.positive[indices],
    )


def split_objective(data: ZoneData, mask: np.ndarray, ratio: float) -> float:
    """二路划分目标函数；holdout 与补集的异常率越接近越好。"""
    hold_obs = data.observed[mask].sum(axis=0)
    hold_pos = data.positive[mask].sum(axis=0)
    remain_obs = data.observed.sum(axis=0) - hold_obs
    remain_pos = data.positive.sum(axis=0) - hold_pos
    total_obs = hold_obs + remain_obs
    total_pos = hold_pos + remain_pos

    valid = (hold_obs > 0) & (remain_obs > 0)
    if np.any(valid):
        hold_rate = hold_pos[valid] / hold_obs[valid]
        remain_rate = remain_pos[valid] / remain_obs[valid]
        pooled = total_pos[valid] / total_obs[valid]
        scale = np.maximum(np.sqrt(pooled * (1.0 - pooled)), 0.02)
        rate_loss = float(np.mean(np.abs(hold_rate - remain_rate) / scale))
    else:
        rate_loss = 0.0

    row_loss = abs(float(data.rows[mask].sum()) / float(data.rows.sum()) - ratio) / max(
        ratio, EPS
    )
    measurable = total_obs > 0
    observed_loss = (
        float(np.mean(np.abs(hold_obs[measurable] / total_obs[measurable] - ratio)))
        / max(ratio, EPS)
        if np.any(measurable)
        else 0.0
    )

    positive_hospitals = np.count_nonzero(data.positive > 0, axis=0)
    negative_hospitals = np.count_nonzero((data.observed - data.positive) > 0, axis=0)
    missing_pos = (positive_hospitals >= 2) & ((hold_pos == 0) | (remain_pos == 0))
    hold_neg = hold_obs - hold_pos
    remain_neg = remain_obs - remain_pos
    missing_neg = (negative_hospitals >= 2) & ((hold_neg == 0) | (remain_neg == 0))
    missing_obs = (np.count_nonzero(data.observed > 0, axis=0) >= 2) & ~valid
    constraint_loss = float(missing_pos.sum() + missing_neg.sum() + missing_obs.sum())
    return (
        RATE_WEIGHT * rate_loss
        + ROW_BALANCE_WEIGHT * row_loss
        + OBSERVED_BALANCE_WEIGHT * observed_loss
        + FEASIBILITY_PENALTY * constraint_loss
    )


def all_swap_objectives(
    data: ZoneData,
    hold_indices: np.ndarray,
    remain_indices: np.ndarray,
    hold_rows: float,
    hold_obs: np.ndarray,
    hold_pos: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """向量化计算所有一进一出医院交换的得分。"""
    candidate_obs = (
        hold_obs[None, None, :]
        - data.observed[hold_indices, None, :]
        + data.observed[None, remain_indices, :]
    )
    candidate_pos = (
        hold_pos[None, None, :]
        - data.positive[hold_indices, None, :]
        + data.positive[None, remain_indices, :]
    )
    total_obs = data.observed.sum(axis=0)
    total_pos = data.positive.sum(axis=0)
    remain_obs = total_obs[None, None, :] - candidate_obs
    remain_pos = total_pos[None, None, :] - candidate_pos
    valid = (candidate_obs > 0) & (remain_obs > 0)

    hold_rate = np.divide(
        candidate_pos,
        candidate_obs,
        out=np.zeros_like(candidate_pos),
        where=candidate_obs > 0,
    )
    remain_rate = np.divide(
        remain_pos, remain_obs, out=np.zeros_like(remain_pos), where=remain_obs > 0
    )
    pooled = np.divide(
        total_pos, total_obs, out=np.zeros_like(total_pos), where=total_obs > 0
    )
    scale = np.maximum(np.sqrt(pooled * (1.0 - pooled)), 0.02)
    rate_terms = np.where(valid, np.abs(hold_rate - remain_rate) / scale, 0.0)
    rate_loss = rate_terms.sum(axis=2) / np.maximum(valid.sum(axis=2), 1)

    candidate_rows = (
        hold_rows - data.rows[hold_indices, None] + data.rows[None, remain_indices]
    )
    row_loss = np.abs(candidate_rows / float(data.rows.sum()) - ratio) / max(ratio, EPS)
    measurable = total_obs > 0
    if np.any(measurable):
        observed_share = candidate_obs[:, :, measurable] / total_obs[measurable]
        observed_loss = np.mean(np.abs(observed_share - ratio), axis=2) / max(
            ratio, EPS
        )
    else:
        observed_loss = np.zeros_like(row_loss)

    positive_hospitals = np.count_nonzero(data.positive > 0, axis=0)
    negative_hospitals = np.count_nonzero((data.observed - data.positive) > 0, axis=0)
    missing_pos = (positive_hospitals >= 2) & ((candidate_pos == 0) | (remain_pos == 0))
    candidate_neg = candidate_obs - candidate_pos
    remain_neg = remain_obs - remain_pos
    missing_neg = (negative_hospitals >= 2) & ((candidate_neg == 0) | (remain_neg == 0))
    missing_obs = (np.count_nonzero(data.observed > 0, axis=0) >= 2) & ~valid
    constraint_loss = (
        missing_pos.sum(axis=2) + missing_neg.sum(axis=2) + missing_obs.sum(axis=2)
    )
    return (
        RATE_WEIGHT * rate_loss
        + ROW_BALANCE_WEIGHT * row_loss
        + OBSERVED_BALANCE_WEIGHT * observed_loss
        + FEASIBILITY_PENALTY * constraint_loss
    )


def optimize_holdout(
    data: ZoneData, ratio: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    n_hospitals = len(data.hospids)
    if n_hospitals < 2:
        raise ValueError(f"地区 {data.zone!r} 只有 {n_hospitals} 家医院，无法拆分")
    n_holdout = min(max(int(math.floor(n_hospitals * ratio + 0.5)), 1), n_hospitals - 1)
    best_mask: np.ndarray | None = None
    best_score = math.inf

    for _ in range(N_RESTARTS):
        mask = np.zeros(n_hospitals, dtype=bool)
        mask[rng.choice(n_hospitals, size=n_holdout, replace=False)] = True
        score = split_objective(data, mask, ratio)
        hold_rows = float(data.rows[mask].sum())
        hold_obs = data.observed[mask].sum(axis=0)
        hold_pos = data.positive[mask].sum(axis=0)

        for _ in range(MAX_SWAP_STEPS):
            hold_indices = np.flatnonzero(mask)
            remain_indices = np.flatnonzero(~mask)
            rng.shuffle(hold_indices)
            rng.shuffle(remain_indices)
            scores = all_swap_objectives(
                data, hold_indices, remain_indices, hold_rows, hold_obs, hold_pos, ratio
            )
            flat_index = int(np.argmin(scores))
            hold_pos_i, remain_pos_i = np.unravel_index(flat_index, scores.shape)
            next_score = float(scores[hold_pos_i, remain_pos_i])
            if next_score >= score - 1e-12:
                break
            old_i = int(hold_indices[hold_pos_i])
            new_i = int(remain_indices[remain_pos_i])
            hold_rows += float(data.rows[new_i] - data.rows[old_i])
            hold_obs += data.observed[new_i] - data.observed[old_i]
            hold_pos += data.positive[new_i] - data.positive[old_i]
            mask[old_i] = False
            mask[new_i] = True
            score = next_score

        if score < best_score:
            best_score = score
            best_mask = mask.copy()
    assert best_mask is not None
    return best_mask, best_score


@dataclass
class JointObjectiveContext:
    total_rows: float
    total_observed: np.ndarray
    total_positive: np.ndarray
    feasible_positive: np.ndarray
    feasible_negative: np.ndarray
    feasible_observed: np.ndarray


def make_joint_context(data: ZoneData) -> JointObjectiveContext:
    return JointObjectiveContext(
        total_rows=float(data.rows.sum()),
        total_observed=data.observed.sum(axis=0),
        total_positive=data.positive.sum(axis=0),
        feasible_positive=np.count_nonzero(data.positive > 0, axis=0) >= 3,
        feasible_negative=np.count_nonzero((data.observed - data.positive) > 0, axis=0)
        >= 3,
        feasible_observed=np.count_nonzero(data.observed > 0, axis=0) >= 3,
    )


def joint_objective_from_stats(
    context: JointObjectiveContext,
    group_rows: np.ndarray,
    group_observed: np.ndarray,
    group_positive: np.ndarray,
) -> float:
    """同时评价 train/internal/external 三组；越小越好。"""
    valid = np.all(group_observed > 0, axis=0)
    rates = np.divide(
        group_positive,
        group_observed,
        out=np.zeros_like(group_positive),
        where=group_observed > 0,
    )
    pooled = np.divide(
        context.total_positive,
        context.total_observed,
        out=np.zeros_like(context.total_positive),
        where=context.total_observed > 0,
    )
    scale = np.maximum(np.sqrt(pooled * (1.0 - pooled)), 0.02)
    standardized_gap = (rates.max(axis=0) - rates.min(axis=0)) / scale
    if np.any(valid):
        rate_loss = float(np.mean(standardized_gap[valid]))
        worst_rate_loss = float(np.max(standardized_gap[valid]))
    else:
        rate_loss = 0.0
        worst_rate_loss = 0.0

    row_shares = group_rows / context.total_rows
    row_loss = float(np.mean(np.abs(row_shares - SPLIT_TARGETS) / SPLIT_TARGETS))

    measurable = context.total_observed > 0
    if np.any(measurable):
        observed_shares = (
            group_observed[:, measurable] / context.total_observed[measurable]
        )
        observed_loss = float(
            np.mean(
                np.abs(observed_shares - SPLIT_TARGETS[:, None])
                / SPLIT_TARGETS[:, None]
            )
        )
    else:
        observed_loss = 0.0

    group_negative = group_observed - group_positive
    missing_positive = (
        context.feasible_positive[None, :] & (group_positive == 0)
    ).sum()
    missing_negative = (
        context.feasible_negative[None, :] & (group_negative == 0)
    ).sum()
    missing_observed = (
        context.feasible_observed[None, :] & (group_observed == 0)
    ).sum()
    constraint_loss = float(missing_positive + missing_negative + missing_observed)

    return (
        RATE_WEIGHT * rate_loss
        + WORST_RATE_GAP_WEIGHT * worst_rate_loss
        + ROW_BALANCE_WEIGHT * row_loss
        + OBSERVED_BALANCE_WEIGHT * observed_loss
        + FEASIBILITY_PENALTY * constraint_loss
    )


def hospital_split_counts(n_hospitals: int) -> np.ndarray:
    """内部、外部验证医院数相同，剩余医院全部作为训练集。"""
    n_validation = int(math.floor(n_hospitals * 0.20 + 0.5))
    n_validation = min(max(n_validation, 1), (n_hospitals - 1) // 2)
    n_train = n_hospitals - 2 * n_validation
    return np.asarray([n_train, n_validation, n_validation], dtype=np.int64)


def group_stats_from_labels(
    data: ZoneData, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_rows = np.asarray(
        [data.rows[labels == group].sum() for group in range(3)], dtype=np.float64
    )
    group_observed = np.asarray(
        [data.observed[labels == group].sum(axis=0) for group in range(3)],
        dtype=np.float64,
    )
    group_positive = np.asarray(
        [data.positive[labels == group].sum(axis=0) for group in range(3)],
        dtype=np.float64,
    )
    return group_rows, group_observed, group_positive


def exact_joint_split(data: ZoneData) -> tuple[np.ndarray, float, int]:
    """枚举小地区所有满足医院数约束的三路组合，得到全局最优解。"""
    n = len(data.hospids)
    counts = hospital_split_counts(n)
    n_internal = int(counts[1])
    n_external = int(counts[2])
    context = make_joint_context(data)
    total_rows = float(data.rows.sum())
    total_observed = data.observed.sum(axis=0)
    total_positive = data.positive.sum(axis=0)
    all_indices = tuple(range(n))
    best_labels: np.ndarray | None = None
    best_score = math.inf
    evaluated = 0

    for external_tuple in combinations(all_indices, n_external):
        external_indices = np.asarray(external_tuple, dtype=np.int64)
        external_set = set(external_tuple)
        remaining = tuple(i for i in all_indices if i not in external_set)
        external_rows = float(data.rows[external_indices].sum())
        external_observed = data.observed[external_indices].sum(axis=0)
        external_positive = data.positive[external_indices].sum(axis=0)

        for internal_tuple in combinations(remaining, n_internal):
            internal_indices = np.asarray(internal_tuple, dtype=np.int64)
            internal_rows = float(data.rows[internal_indices].sum())
            internal_observed = data.observed[internal_indices].sum(axis=0)
            internal_positive = data.positive[internal_indices].sum(axis=0)
            group_rows = np.asarray(
                [
                    total_rows - internal_rows - external_rows,
                    internal_rows,
                    external_rows,
                ],
                dtype=np.float64,
            )
            group_observed = np.asarray(
                [
                    total_observed - internal_observed - external_observed,
                    internal_observed,
                    external_observed,
                ]
            )
            group_positive = np.asarray(
                [
                    total_positive - internal_positive - external_positive,
                    internal_positive,
                    external_positive,
                ]
            )
            score = joint_objective_from_stats(
                context, group_rows, group_observed, group_positive
            )
            evaluated += 1
            if score < best_score:
                labels = np.full(n, 0, dtype=np.int8)
                labels[internal_indices] = 1
                labels[external_indices] = 2
                best_labels = labels
                best_score = score

    assert best_labels is not None
    return best_labels, best_score, evaluated


def refine_joint_split(
    data: ZoneData,
    initial_labels: np.ndarray,
    context: JointObjectiveContext,
) -> tuple[np.ndarray, float]:
    """在三组之间交换医院，直到任意一次交换都不能继续降低目标。"""
    labels = initial_labels.copy()
    group_rows, group_observed, group_positive = group_stats_from_labels(data, labels)
    score = joint_objective_from_stats(
        context, group_rows, group_observed, group_positive
    )

    for _ in range(JOINT_MAX_SWAP_STEPS):
        best_score = score
        best_swap: tuple[int, int] | None = None
        best_stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for group_a, group_b in ((0, 1), (0, 2), (1, 2)):
            indices_a = np.flatnonzero(labels == group_a)
            indices_b = np.flatnonzero(labels == group_b)
            for index_a in indices_a:
                for index_b in indices_b:
                    rows = group_rows.copy()
                    observed = group_observed.copy()
                    positive = group_positive.copy()
                    rows[group_a] += data.rows[index_b] - data.rows[index_a]
                    rows[group_b] += data.rows[index_a] - data.rows[index_b]
                    observed[group_a] += data.observed[index_b] - data.observed[index_a]
                    observed[group_b] += data.observed[index_a] - data.observed[index_b]
                    positive[group_a] += data.positive[index_b] - data.positive[index_a]
                    positive[group_b] += data.positive[index_a] - data.positive[index_b]
                    candidate_score = joint_objective_from_stats(
                        context, rows, observed, positive
                    )
                    if candidate_score < best_score - 1e-12:
                        best_score = candidate_score
                        best_swap = (int(index_a), int(index_b))
                        best_stats = (rows, observed, positive)

        if best_swap is None or best_stats is None:
            break
        index_a, index_b = best_swap
        labels[index_a], labels[index_b] = labels[index_b], labels[index_a]
        group_rows, group_observed, group_positive = best_stats
        score = best_score

    return labels, score


def heuristic_joint_split(
    data: ZoneData, rng: np.random.Generator
) -> tuple[np.ndarray, float, int]:
    """大地区采用多次随机初始化加三组联合交换。"""
    counts = hospital_split_counts(len(data.hospids))
    base_labels = np.concatenate(
        [np.full(int(counts[group]), group, dtype=np.int8) for group in range(3)]
    )
    context = make_joint_context(data)
    best_labels: np.ndarray | None = None
    best_score = math.inf

    for _ in range(JOINT_HEURISTIC_RESTARTS):
        initial_labels = base_labels[rng.permutation(len(base_labels))]
        labels, score = refine_joint_split(data, initial_labels, context)
        if score < best_score:
            best_labels = labels
            best_score = score
    assert best_labels is not None
    return best_labels, best_score, JOINT_HEURISTIC_RESTARTS


def split_zone_three_way(
    data: ZoneData, rng: np.random.Generator
) -> tuple[np.ndarray, float, str]:
    """对三个集合进行联合优化，而不是依次进行两个二路优化。"""
    n = len(data.hospids)
    if n < 5:
        raise ValueError(f"地区 {data.zone!r} 只有 {n} 家医院，无法稳定按 3:1:1 划分")
    if n <= EXACT_ENUMERATION_MAX_HOSPITALS:
        integer_labels, score, evaluated = exact_joint_split(data)
        method = f"exact_joint（枚举 {evaluated} 个组合）"
    else:
        integer_labels, score, restarts = heuristic_joint_split(data, rng)
        method = f"joint_swap（{restarts} 次初始化）"
    return SPLIT_NAMES[integer_labels], score, method


def print_quality_report(data: ZoneData, labels: np.ndarray) -> None:
    print(f"\n[{data.zone}] 指标异常率")
    print(
        f"{'indicator':<16} {'train':>12} {'internal':>12} {'external':>12} {'max_diff':>12}"
    )
    for j, indicator in enumerate(INDICATORS):
        rates: list[float | None] = []
        displays = []
        for split in (TRAIN, INTERNAL, EXTERNAL):
            mask = labels == split
            observed = int(data.observed[mask, j].sum())
            positive = int(data.positive[mask, j].sum())
            rate = positive / observed if observed else None
            rates.append(rate)
            displays.append("NA" if rate is None else f"{rate:.4%}")
        valid_rates = [rate for rate in rates if rate is not None]
        max_diff = (
            max(valid_rates) - min(valid_rates) if len(valid_rates) >= 2 else None
        )
        diff_text = "NA" if max_diff is None else f"{max_diff:.4%}"
        print(
            f"{indicator:<16} {displays[0]:>12} {displays[1]:>12} {displays[2]:>12} {diff_text:>12}"
        )


def output_table_exists(conn: pymysql.connections.Connection) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*) AS n FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
            """,
            (OUTPUT_TABLE,),
        )
        return int(cursor.fetchone()["n"]) > 0


def create_output_table(
    conn: pymysql.connections.Connection,
    sampled_rows: list[dict[str, Any]],
    hospital_to_split: dict[tuple[str, int], str],
) -> None:
    """复制源表结构和入选行，并增加 split 列。"""
    selected = [
        (
            str(row["uid"]),
            str(row["id"]),
            hospital_to_split[(str(row["economic_zone"]), int(row["hospid"]))],
        )
        for row in sampled_rows
    ]
    if len(selected) != TARGET_DATASET_ROWS:
        raise AssertionError("写表前的入选行数不等于 TARGET_DATASET_ROWS")

    created_output = False
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TEMPORARY TABLE tmp_selected_dataset_rows (
                    uid CHAR(32) NOT NULL,
                    id CHAR(18) NOT NULL,
                    split ENUM('train','internal_validation','external_validation') NOT NULL,
                    PRIMARY KEY (uid, id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
            for start in range(0, len(selected), 5_000):
                cursor.executemany(
                    "INSERT INTO tmp_selected_dataset_rows (uid,id,split) VALUES (%s,%s,%s)",
                    selected[start : start + 5_000],
                )

            cursor.execute(f"CREATE TABLE `{OUTPUT_TABLE}` LIKE `{SOURCE_TABLE}`")
            created_output = True
            cursor.execute(f"""
                ALTER TABLE `{OUTPUT_TABLE}`
                ADD COLUMN split ENUM('train','internal_validation','external_validation') NOT NULL,
                ADD KEY idx_split (split)
                """)
            cursor.execute(f"""
                INSERT INTO `{OUTPUT_TABLE}`
                SELECT source_row.*, selected_row.split
                FROM `{SOURCE_TABLE}` AS source_row
                JOIN tmp_selected_dataset_rows AS selected_row
                  ON source_row.uid = selected_row.uid AND source_row.id = selected_row.id
                """)
            cursor.execute(f"SELECT COUNT(*) AS n FROM `{OUTPUT_TABLE}`")
            actual = int(cursor.fetchone()["n"])
            if actual != TARGET_DATASET_ROWS:
                raise RuntimeError(
                    f"输出表实际写入 {actual} 行，不等于目标 {TARGET_DATASET_ROWS} 行"
                )
        conn.commit()
    except Exception:
        conn.rollback()
        if created_output:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE `{OUTPUT_TABLE}`")
            conn.commit()
        raise


def main() -> None:
    conn = pymysql.connect(cursorclass=DictCursor, **DB_CONFIG)
    try:
        if output_table_exists(conn):
            raise RuntimeError(
                f"输出表 `{OUTPUT_TABLE}` 已存在。请修改 OUTPUT_TABLE，"
                "或确认旧表无用后手动 DROP TABLE。"
            )

        eligible_rows = fetch_eligible_rows(conn)
        if not eligible_rows:
            raise RuntimeError("过滤后没有可用于划分的数据")
        rng = np.random.default_rng(RANDOM_SEED)
        sampled_rows = proportional_sample_by_hospital(
            eligible_rows, TARGET_DATASET_ROWS, rng
        )
        print(
            f"排除 {EXCLUDED_IMAGE_PREFIX} 且去掉地区/医院为空的数据后共有 "
            f"{len(eligible_rows)} 行；已抽取 {len(sampled_rows)} 行。"
        )

        zones = build_zone_data(sampled_rows)
        hospital_to_split: dict[tuple[str, int], str] = {}
        for data in zones:
            labels, joint_score, optimization_method = split_zone_three_way(data, rng)
            print(f"\n[{data.zone}] 划分结果")
            for split in (TRAIN, INTERNAL, EXTERNAL):
                mask = labels == split
                print(
                    f"  {split:<20} 医院 {int(mask.sum()):>4} 家，数据 {int(data.rows[mask].sum()):>6} 行"
                )
            print(
                f"  optimization: {optimization_method}, "
                f"joint_score={joint_score:.6f}"
            )
            for hospid, split in zip(data.hospids, labels):
                hospital_to_split[(data.zone, int(hospid))] = str(split)
            print_quality_report(data, labels)

        expected_hospitals = {
            (str(row["economic_zone"]), int(row["hospid"])) for row in sampled_rows
        }
        if set(hospital_to_split) != expected_hospitals:
            raise AssertionError("医院划分映射不完整")

        create_output_table(conn, sampled_rows, hospital_to_split)
        print(f"\n完成：`{OUTPUT_TABLE}` 已生成，共 {TARGET_DATASET_ROWS} 行。")
        print("split 取值：train、internal_validation、external_validation")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
