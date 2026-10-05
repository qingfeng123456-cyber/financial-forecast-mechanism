from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


EPS = 1e-8


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    out = np.exp(np.clip(shifted, -40.0, 40.0))
    return out / np.maximum(out.sum(axis=-1, keepdims=True), EPS)


def _qlike(log_forecast: np.ndarray, target: np.ndarray) -> np.ndarray:
    return log_forecast + target[..., None] * np.exp(-log_forecast)


def stack_records(records_by_model: dict[str, list[dict]], names: list[str]) -> list[dict]:
    """Align frozen forecasts and verify that every method sees identical rows."""
    reference = records_by_model[names[0]]
    if any(len(records_by_model[name]) != len(reference) for name in names):
        raise ValueError("Base forecast record counts differ")
    rows = []
    for position, ref in enumerate(reference):
        forecasts = []
        for name in names:
            item = records_by_model[name][position]
            if item["date"] != ref["date"] or item["date_index"] != ref["date_index"]:
                raise ValueError("Base forecast dates are not aligned")
            if not np.array_equal(item["assets"], ref["assets"]):
                raise ValueError("Base forecast asset universes are not aligned")
            if not np.allclose(item["target"], ref["target"], equal_nan=True):
                raise ValueError("Base forecast targets are not aligned")
            forecasts.append(np.asarray(item["forecast"], dtype=np.float64))
        matrix = np.column_stack(forecasts)
        if not np.isfinite(matrix).all() or np.any(matrix <= 0):
            raise ValueError("Aggregation received an invalid base forecast")
        rows.append({"date": ref["date"], "date_index": int(ref["date_index"]),
                     "assets": np.asarray(ref["assets"], dtype=np.int64),
                     "target": np.asarray(ref["target"], dtype=np.float64),
                     "log_forecasts": np.log(matrix), "stress": float(ref["stress"])})
    return rows


def fit_static_weights(rows: list[dict], n_models: int) -> tuple[np.ndarray, dict]:
    """Fit one validation-only log-scale QLIKE pool with equal date weights."""
    def objective(logits: np.ndarray) -> float:
        weights = _softmax(logits)
        scores = []
        for row in rows:
            good = np.isfinite(row["target"])
            if int(good.sum()) < 8:
                continue
            z = row["log_forecasts"][good] @ weights
            y = row["target"][good]
            scores.append(float(np.mean(z + y * np.exp(-z))))
        if not scores:
            raise ValueError("No validation rows for static aggregation")
        return float(np.mean(scores))

    result = minimize(objective, np.zeros(n_models), method="L-BFGS-B",
                      bounds=[(-6.0, 6.0)] * n_models,
                      options={"maxiter": 200, "ftol": 1e-10})
    if not np.isfinite(result.fun):
        raise RuntimeError("Static QLIKE pool did not produce a finite objective")
    record = {"converged": bool(result.success), "message": str(result.message),
              "iterations": int(result.nit), "objective": float(result.fun)}
    return _softmax(result.x), record


def estimate_kappa(rows: list[dict], n_assets: int, decay: float, clip: float) -> float:
    """Validation-only moment estimate of local-performance signal versus noise."""
    n_models = rows[0]["log_forecasts"].shape[1]
    sums = np.zeros((n_assets, n_models), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    weights = np.zeros(n_assets, dtype=np.float64)
    weights_sq = np.zeros(n_assets, dtype=np.float64)
    for row in rows:
        sums *= decay
        sums_sq *= decay
        weights *= decay
        weights_sq *= decay * decay
        good = np.isfinite(row["target"])
        if not good.any():
            continue
        assets = row["assets"][good]
        loss = _qlike(row["log_forecasts"][good], row["target"][good])
        centered = np.clip(loss - loss.mean(axis=1, keepdims=True), -clip, clip)
        sums[assets] += centered
        sums_sq[assets] += centered * centered
        weights[assets] += 1.0
        weights_sq[assets] += 1.0
    seen = weights > 2.0
    if int(seen.sum()) < 8:
        return 1e12
    means = sums[seen] / weights[seen, None]
    global_mean = np.average(means, axis=0, weights=weights[seen])
    between = float(np.mean(np.square(means - global_mean)))
    variances = np.maximum(sums_sq[seen] / weights[seen, None] - means * means, 0.0)
    n_eff = np.square(weights[seen]) / np.maximum(weights_sq[seen], EPS)
    estimation_noise = float(np.mean(variances / np.maximum(n_eff[:, None], 1.0)))
    signal = max(between - estimation_noise, 0.0)
    if signal <= 1e-10:
        return 1e12
    reference_n = float(np.median(n_eff))
    return float(np.clip(reference_n * estimation_noise / signal, 4.0, 1e6))


@dataclass
class MemoryState:
    decay: float
    n_assets: int
    n_models: int
    prior: np.ndarray
    temperature: float
    clip: float

    def __post_init__(self):
        self.global_sum = np.zeros(self.n_models, dtype=np.float64)
        self.global_weight = 0.0
        self.local_sum = np.zeros((self.n_assets, self.n_models), dtype=np.float64)
        self.local_weight = np.zeros(self.n_assets, dtype=np.float64)
        self.local_weight_sq = np.zeros(self.n_assets, dtype=np.float64)

    def update(self, assets: np.ndarray, log_forecasts: np.ndarray, target: np.ndarray) -> None:
        self.global_sum *= self.decay
        self.global_weight *= self.decay
        self.local_sum *= self.decay
        self.local_weight *= self.decay
        self.local_weight_sq *= self.decay * self.decay
        good = np.isfinite(target)
        if not good.any():
            return
        assets = assets[good]
        loss = _qlike(log_forecasts[good], target[good])
        centered = np.clip(loss - loss.mean(axis=1, keepdims=True), -self.clip, self.clip)
        self.global_sum += centered.mean(axis=0)
        self.global_weight += 1.0
        self.local_sum[assets] += centered
        self.local_weight[assets] += 1.0
        self.local_weight_sq[assets] += 1.0

    def weights(self, assets: np.ndarray, kappa: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prior_logits = np.log(np.maximum(self.prior, EPS))
        global_mean = self.global_sum / max(self.global_weight, 1.0)
        global_logits = prior_logits - self.temperature * global_mean
        n_eff = np.divide(np.square(self.local_weight[assets]), self.local_weight_sq[assets],
                          out=np.zeros(len(assets)), where=self.local_weight_sq[assets] > 0)
        if kappa is None:
            weights = np.broadcast_to(_softmax(global_logits), (len(assets), self.n_models)).copy()
            return weights, np.zeros(len(assets), dtype=np.float64), n_eff
        local_mean = np.divide(self.local_sum[assets], self.local_weight[assets, None],
                               out=np.zeros((len(assets), self.n_models)),
                               where=self.local_weight[assets, None] > 0)
        local_logits = prior_logits[None, :] - self.temperature * local_mean
        if float(kappa) == 0.0:
            rho = (n_eff > 0).astype(np.float64)
        elif float(kappa) >= 1e11:
            rho = np.zeros_like(n_eff)
        else:
            rho = n_eff / (n_eff + float(kappa))
        shrunk = (1.0 - rho[:, None]) * global_logits[None, :] + rho[:, None] * local_logits
        return _softmax(shrunk), rho, n_eff

    def predict(self, assets: np.ndarray, log_forecasts: np.ndarray,
                kappa: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weights, rho, n_eff = self.weights(assets, kappa)
        return np.sum(weights * log_forecasts, axis=1), weights, rho, n_eff


class MetaState:
    def __init__(self, n_components: int, decay: float, temperature: float, clip: float):
        self.decay, self.temperature, self.clip = decay, temperature, clip
        self.sum = np.zeros(n_components, dtype=np.float64)
        self.weight = 0.0

    def weights(self) -> np.ndarray:
        mean = self.sum / max(self.weight, 1.0)
        return _softmax(-self.temperature * mean)

    def update(self, component_log_forecasts: np.ndarray, target: np.ndarray) -> None:
        self.sum *= self.decay
        self.weight *= self.decay
        good = np.isfinite(target)
        if not good.any():
            return
        loss = _qlike(component_log_forecasts[good], target[good])
        centered = np.clip(loss - loss.mean(axis=1, keepdims=True), -self.clip, self.clip)
        self.sum += centered.mean(axis=0)
        self.weight += 1.0


class DashVReplay:
    def __init__(self, n_assets: int, n_models: int, static_weights: np.ndarray,
                 kappas: dict[float, float], cfg):
        self.cfg = cfg
        self.static = static_weights
        decays = sorted(set([cfg.single_memory_decay, *cfg.memory_decays]))
        self.states = {decay: MemoryState(decay, n_assets, n_models, static_weights,
                                          cfg.aggregation_temperature,
                                          cfg.aggregation_loss_clip) for decay in decays}
        self.kappas = kappas
        self.meta_pooled = MetaState(len(cfg.memory_decays) + 1, cfg.single_memory_decay,
                                     cfg.meta_temperature, cfg.aggregation_loss_clip)
        self.meta_hier = MetaState(len(cfg.memory_decays) + 1, cfg.single_memory_decay,
                                   cfg.meta_temperature, cfg.aggregation_loss_clip)
        self.previous_effective = None

    def issue(self, row: dict) -> tuple[dict[str, np.ndarray], dict]:
        assets, logs = row["assets"], row["log_forecasts"]
        static_log = logs @ self.static
        single = self.states[self.cfg.single_memory_decay]
        pooled_single, _, _, _ = single.predict(assets, logs, None)
        hier_single, hier_weights, hier_rho, _ = single.predict(
            assets, logs, self.kappas[self.cfg.single_memory_decay])
        local_single, _, _, _ = single.predict(assets, logs, 0.0)

        pooled_parts, hier_parts = [], []
        hier_base_weights, rhos, effective_sizes = [], [], []
        for decay in self.cfg.memory_decays:
            state = self.states[decay]
            pooled, _, _, _ = state.predict(assets, logs, None)
            hierarchical, weights, rho, n_eff = state.predict(
                assets, logs, self.kappas[decay])
            pooled_parts.append(pooled)
            hier_parts.append(hierarchical)
            hier_base_weights.append(weights)
            rhos.append(rho)
            effective_sizes.append(n_eff)
        pooled_components = np.column_stack([*pooled_parts, static_log])
        hier_components = np.column_stack([*hier_parts, static_log])
        meta_pooled = self.meta_pooled.weights()
        meta_hier = self.meta_hier.weights()
        multi_log = pooled_components @ meta_pooled
        full_log = hier_components @ meta_hier

        effective = meta_hier[-1] * self.static[None, :]
        for weight, component_weights in zip(meta_hier[:-1], hier_base_weights):
            effective = effective + weight * component_weights
        average_effective = effective.mean(axis=0)
        turnover = (float(np.abs(average_effective - self.previous_effective).sum())
                    if self.previous_effective is not None else 0.0)
        self.previous_effective = average_effective.copy()
        diagnostics = {
            "meta_fast": float(meta_hier[0]), "meta_medium": float(meta_hier[1]),
            "meta_slow": float(meta_hier[2]), "meta_static": float(meta_hier[3]),
            "rho_fast": float(np.mean(rhos[0])), "rho_medium": float(np.mean(rhos[1])),
            "rho_slow": float(np.mean(rhos[2])), "weight_turnover": turnover,
            "n_eff_fast": float(np.mean(effective_sizes[0])),
            "n_eff_medium": float(np.mean(effective_sizes[1])),
            "n_eff_slow": float(np.mean(effective_sizes[2])),
            "full_vs_static_abs_log": float(np.mean(np.abs(full_log - static_log))),
            "forecast_dispersion": float(np.std(np.exp(full_log))),
            **{f"base_weight_{j}": float(value) for j, value in enumerate(average_effective)},
        }
        predictions = {
            "equal_geometric": np.mean(logs, axis=1),
            "equal_arithmetic": np.log(np.maximum(np.mean(np.exp(logs), axis=1), EPS)),
            "static_qlike": static_log,
            "asset_local_single": local_single,
            "dashv_base": pooled_single,
            "dashv_hier": hier_single,
            "dashv_multi": multi_log,
            "dashv_full": full_log,
        }
        issued = {"assets": assets, "log_forecasts": logs, "target": row["target"],
                  "date_index": row["date_index"], "pooled_components": pooled_components,
                  "hier_components": hier_components}
        return predictions, {"issued": issued, "diagnostics": diagnostics}

    def update(self, issued: dict) -> None:
        for state in self.states.values():
            state.update(issued["assets"], issued["log_forecasts"], issued["target"])
        self.meta_pooled.update(issued["pooled_components"], issued["target"])
        self.meta_hier.update(issued["hier_components"], issued["target"])


def run_aggregation(valid_rows: list[dict], test_rows: list[dict], n_assets: int,
                    base_names: list[str], cfg) -> tuple[dict[str, list[dict]], list[dict], dict]:
    n_models = len(base_names)
    static, static_fit = fit_static_weights(valid_rows, n_models)
    kappas = {decay: estimate_kappa(valid_rows, n_assets, decay, cfg.aggregation_loss_clip)
              for decay in sorted(set([cfg.single_memory_decay, *cfg.memory_decays]))}
    replay = DashVReplay(n_assets, n_models, static, kappas, cfg)

    def mature(pending: list[dict], current_index: int | None) -> None:
        while pending and (current_index is None or
                           pending[0]["date_index"] + cfg.horizon <= current_index):
            replay.update(pending.pop(0))

    pending = []
    for row in valid_rows:
        mature(pending, row["date_index"])
        _, payload = replay.issue(row)
        pending.append(payload["issued"])
    mature(pending, None)

    output = {name: [] for name in [*cfg.aggregation_controls, *cfg.factorial_cells]}
    diagnostics = []
    pending = []
    for row in test_rows:
        mature(pending, row["date_index"])
        predictions, payload = replay.issue(row)
        pending.append(payload["issued"])
        for name, log_prediction in predictions.items():
            output[name].append({"date": row["date"], "date_index": row["date_index"],
                                 "assets": row["assets"], "target": row["target"],
                                 "forecast": np.exp(log_prediction),
                                 "stress": row["stress"]})
        diagnostics.append({"date": row["date"], "date_index": row["date_index"],
                            "n_assets": int(len(row["assets"])), "stress": row["stress"],
                            **payload["diagnostics"]})
    fit = {"base_models": base_names, "static_weights": static.tolist(),
           "static_optimizer": static_fit,
           "memory_kappa": {str(key): value for key, value in kappas.items()},
           "memory_decays": cfg.memory_decays,
           "single_memory_decay": cfg.single_memory_decay}
    return output, diagnostics, fit
