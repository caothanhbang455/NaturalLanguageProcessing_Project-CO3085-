"""Tinh chỉnh siêu tham số trên tập validation (dev).

Chiến lược hai tầng để tiết kiệm tính toán:

1. **Sàng lọc** – chạy toàn bộ lưới M x N với tham số mặc định
   (:meth:`ExperimentRunner.run_grid`).
2. **Tinh chỉnh** – chỉ các cấu hình top-k mới được dò lưới siêu tham số
   (:func:`tune_top_k`).

Dev được dùng để chọn tham số (đúng vai trò validation). Vì chọn trên dev nên điểm dev
của cấu hình tốt nhất sẽ hơi lạc quan – lý do phải giữ test hoàn toàn riêng.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler

from absa.config import PRIMARY_METRIC, SEED
from absa.models.ml.registry import get_model_spec
from absa.training.experiment import ExperimentRunner


def candidate_params(
    param_grid: Mapping[str, list], max_candidates: int = 12, seed: int = SEED
) -> list[dict]:
    """Sinh danh sách bộ tham số: cả lưới nếu đủ nhỏ, ngược lại lấy mẫu ngẫu nhiên."""
    if not param_grid:
        return [{}]
    grid = list(ParameterGrid(param_grid))
    if len(grid) <= max_candidates:
        return grid
    return list(ParameterSampler(param_grid, n_iter=max_candidates, random_state=seed))


def tune_model(
    runner: ExperimentRunner,
    feature: str,
    model: str,
    strategy: str = "multilabel",
    param_grid: Mapping[str, list] | None = None,
    max_candidates: int = 12,
    metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Dò siêu tham số cho một cặp (biểu diễn, mô hình).

    Returns:
        Bảng các bộ tham số đã thử, sắp xếp giảm dần theo ``metric``.
    """
    grid = param_grid if param_grid is not None else get_model_spec(model).param_grid
    rows = [
        runner.run(feature, model, strategy, params, tag="tuning")
        for params in candidate_params(grid, max_candidates)
    ]
    frame = pd.DataFrame(rows)
    frame = frame[frame["status"] == "ok"]
    return frame.sort_values(metric, ascending=False).reset_index(drop=True)


def tune_top_k(
    runner: ExperimentRunner,
    k: int = 5,
    metric: str = PRIMARY_METRIC,
    max_candidates: int = 12,
    exclude_models: tuple[str, ...] = ("majority_baseline",),
) -> pd.DataFrame:
    """Tinh chỉnh ``k`` cấu hình tốt nhất của vòng sàng lọc.

    Returns:
        Mỗi cấu hình một dòng: điểm trước/sau tinh chỉnh và bộ tham số tốt nhất.
    """
    board = runner.leaderboard(metric, top=None, tag="grid")
    board = board[~board["model"].isin(exclude_models)].head(k)
    summary = []
    for _, row in board.iterrows():
        tuned = tune_model(
            runner,
            row["feature"],
            row["model"],
            row["strategy"],
            max_candidates=max_candidates,
            metric=metric,
        )
        best = tuned.iloc[0] if not tuned.empty else row
        summary.append(
            {
                "feature": row["feature"],
                "model": row["model"],
                "strategy": row["strategy"],
                f"{metric}_default": row[metric],
                f"{metric}_tuned": best[metric],
                "gain": best[metric] - row[metric],
                "best_params": best["params"],
                "n_candidates": len(tuned),
            }
        )
    return (
        pd.DataFrame(summary)
        .sort_values(f"{metric}_tuned", ascending=False)
        .reset_index(drop=True)
    )
