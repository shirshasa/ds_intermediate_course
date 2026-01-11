import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils import calculate_psi, _rolling_mad, _rolling_quantile, _rolling_slope, add_cusum_features


# 2. Drift detectors
class DriftDetector:
    def __init__(self, reference_window_size: int = 24 * 7):
        self.ref_size = int(reference_window_size)

    def ks_test(self, reference, current, threshold: float = 0.05):
        """Kolmogorov-Smirnov test"""
        res = stats.ks_2samp(reference, current)
        # scipy typing differs across versions: support both tuple-like and object-like results
        p_value = getattr(res, "pvalue", None)
        if p_value is None:
            p_value = res[1]
        return bool(p_value < threshold), float(p_value)  # type: ignore

    def psi(self, reference, current, bins: int = 10, threshold: float = 0.2):
        """Population Stability Index"""
        try:
            psi = calculate_psi(
                expected=current,
                actual=reference,
                buckets=bins,
            )
        except ValueError:
            return False, None

        return bool(psi > threshold), float(psi)

    def detect(
        self,
        data,
        current_idx: int,
        window_size: int = 100,
        *,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        psi_bins: int = 10,
        mde: Optional[float] = None,
        verbose: bool = False,
    ):
        """Main detection method.

        Compares two windows:
        - reference: [current_idx - window_size - ref_size, current_idx - window_size)
        - current:   [current_idx - window_size, current_idx)

        Returns None if there is not enough history.
        """

        if current_idx is None:
            raise ValueError("current_idx must be provided")

        x = np.asarray(data)

        end = current_idx
        cur_start = end - window_size
        ref_end = cur_start
        ref_start = ref_end - self.ref_size

        if ref_start < 0 or cur_start < 0 or end > len(x):
            return None

        reference = x[ref_start:ref_end]
        current = x[cur_start:end]

        ks_drift, ks_p_value = self.ks_test(reference, current, threshold=ks_threshold)
        psi_drift, psi_value = self.psi(reference, current, bins=psi_bins, threshold=psi_threshold)

        drift = bool(ks_drift or psi_drift)

        if mde is not None:
            drift = drift and (((np.mean(current) - np.mean(reference)) / np.mean(current)) > mde)

        if verbose and drift:
            print(f"Drift detected: {drift}")
            print(f"Reference mean: {np.mean(reference)}")
            print(f"Current mean: {np.mean(current)}")
            print(f"Delta:", (np.mean(current) - np.mean(reference)) / np.mean(current))
            print(f"KS drift: {ks_drift}")
            print(f"PSI drift: {psi_drift}")

        return {
            "reference_range": (ref_start, ref_end),
            "current_range": (cur_start, end),
            "ks": {
                "drift": ks_drift,
                "p_value": ks_p_value,
                "threshold": ks_threshold,
            },
            "psi": {
                "value": psi_value,
                "threshold": psi_threshold,
                "drift": psi_drift,
            },
            "drift": drift,
        }


@dataclass
class MetricConfig:
    name: str
    period: int = 24
    window_size: int = 24
    reference_window_size: int = 7 * 24
    ks_threshold: float = 0.01
    psi_threshold: float = 0.2
    psi_bins: int = 10

    # Alert layer
    persistence: int = 3
    cooldown: int = 24

    # Decision rule
    min_features_to_alert: int = 1
    features: Optional[List[str]] = None
    mde: Optional[float] = None

    # --- Trend drift (not used) ---
    detect_trend_drift: bool = False

    # Slope feature construction
    trend_rolling_window: Optional[int] = None   # rolling window for slope; default -> window_size
    trend_source: str = "value"               # "value" or "roll_mean" (if available)
    trend_window_size: Optional[int] = None            # current window on slope; default -> window_size
    trend_reference_window_size: Optional[int] = None  # reference size; default -> reference_window_size
    trend_ks_threshold: Optional[float] = None
    trend_psi_threshold: Optional[float] = None
    trend_psi_bins: Optional[int] = None
    trend_mde: Optional[float] = None
    trend_persistence: Optional[int] = None
    trend_cooldown: Optional[int] = None


def build_features(
    series: pd.Series,
    *,
    period: int,
    window_size: int,
    reference_window_size: int,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Baseline features on raw series values (no seasonal-adjust)."""
    s = series.astype(float).copy()

    df = pd.DataFrame(index=s.index)
    df["value"] = s
    df["delta_1"] = s.diff(1)
    df["delta_season"] = s.diff(period)

    # seasonal naive forecast and residual
    df["forecast_seasonal_naive"] = s.shift(period)
    df["one_step_resid"] = s - df["forecast_seasonal_naive"]

    base = df["value"]

    # Rolling robust stats
    df["roll_median"] = base.rolling(window_size).median()
    df["roll_mean"] = base.rolling(window_size).mean()
    df["roll_mad"] = _rolling_mad(base, window_size)

    # Rolling quantiles
    df["q05"] = _rolling_quantile(base, window_size, 0.05)
    df["q95"] = _rolling_quantile(base, window_size, 0.95)
    df["q95_q05_spread"] = df["q95"] - df["q05"]

    # Rolling variance/std
    df["roll_var"] = base.rolling(window_size).var(ddof=1)
    df["roll_std"] = base.rolling(window_size).std(ddof=1)

    # Ratios: current stats / reference stats
    cur_var = base.rolling(window_size).var(ddof=1)
    ref_var = base.shift(window_size).rolling(reference_window_size).var(ddof=1)
    df["var_ratio"] = cur_var / (ref_var + eps)

    cur_mad = _rolling_mad(base, window_size)
    ref_mad = _rolling_mad(base.shift(window_size), reference_window_size)
    df["mad_ratio"] = cur_mad / (ref_mad + eps)

    # CUSUM
    df = add_cusum_features(df, window_size=window_size)

    return df


def build_trend_features(feat_df: pd.DataFrame, rolling_window: int, trend_source: str,) -> pd.DataFrame:
    """Builds slope-based features for a single metric. No alerting here."""

    if trend_source == "roll_mean" and "roll_mean" in feat_df.columns:
        src = feat_df["roll_mean"].astype(float)
    else:
        src = feat_df["value"].astype(float)

    out = pd.DataFrame(index=feat_df.index)
    out["value"] = feat_df["value"].astype(float)
    out["trend_source"] = src
    out["trend_slope"] = _rolling_slope(src, rolling_window)

    alpha = 0.1
    out["trend_slope"] = out["trend_slope"].ewm(alpha=alpha, adjust=False).mean()
    # optional: slope acceleration (often useful for "trend changed")
    out["trend_slope_delta"] = out["trend_slope"].diff(rolling_window)
    out.attrs["trend_rolling_window"] = rolling_window
    out.attrs["trend_source"] = trend_source
    return out


class AlertSystem:
    """Baseline AlertSystem: feature engineering + KS/PSI tests."""

    DEFAULT_FEATURES = [
        "value",
        "delta_season",
        "one_step_resid",
        "roll_median",
        "roll_mean",
        "roll_mad",
        "q05",
        "q95",
        "q95_q05_spread",
        "roll_var",
        "roll_std",
        "var_ratio",
        "mad_ratio",
        "cusum_pos",
        "cusum_var_pos",
    ]

    def __init__(self, metric_configs: Dict[str, MetricConfig], *, timestamp_col: str = "timestamp"):
        self.metric_configs = metric_configs
        self.timestamp_col = timestamp_col

        # Drift detectors per metric (main stream)
        self.detectors: Dict[str, DriftDetector] = {}
        for name, cfg in metric_configs.items():
            self.detectors[name] = DriftDetector(reference_window_size=cfg.reference_window_size)

        # state for persistence/cooldown (main stream)
        self._state: Dict[str, Dict[str, int]] = {name: {"run": 0, "cooldown": 0} for name in metric_configs.keys()}

        # separate state for trend drift stream
        self._trend_state: Dict[str, Dict[str, int]] = {name: {"run": 0, "cooldown": 0} for name in metric_configs.keys()}

        # cache of computed features
        self._features: Dict[str, pd.DataFrame] = {}
        self._trend_features: Dict[str, pd.DataFrame] = {}

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        if self.timestamp_col in df2.columns:
            df2[self.timestamp_col] = pd.to_datetime(df2[self.timestamp_col])
            df2 = df2.sort_values(self.timestamp_col).reset_index(drop=True)
            df2 = df2.set_index(self.timestamp_col)
        else:
            df2 = df2.sort_index()
        return df2

    def compute_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df2 = self._prepare_df(df)
        feats: Dict[str, pd.DataFrame] = {}
        for metric, cfg in self.metric_configs.items():
            if metric not in df2.columns:
                raise KeyError(f"Metric '{metric}' not found in df columns.")
            feats[metric] = build_features(
                df2[metric],
                period=cfg.period,
                window_size=cfg.window_size,
                reference_window_size=cfg.reference_window_size,
            )
        self._features = feats
        return feats

    def compute_trend_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Computes and caches trend/slope features for metrics with detect_trend_drift=True."""
        if not self._features:
            feats = self.compute_features(df)
        else:
            feats = self._features

        trend_feats: Dict[str, pd.DataFrame] = {}
        for metric, cfg in self.metric_configs.items():
            if not cfg.detect_trend_drift:
                continue
            trend_feats[metric] = build_trend_features(feats[metric], cfg.trend_rolling_window or cfg.window_size, cfg.trend_source )
        self._trend_features = trend_feats
        return trend_feats


    # -------------------------
    # Trend drift (separate stream, not used, experimental)
    # -------------------------

    def detect_trend_drift(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Detects trend (slope) drift per metric and returns a separate dataframe.
        """

        trend_feats = self.compute_trend_features(df)
        out_rows: List[Dict[str, Any]] = []

        for metric, tdf in trend_feats.items():
            cfg = self.metric_configs[metric]

            # detection parameters
            cur_w = cfg.trend_window_size or cfg.window_size
            ref_w = cfg.trend_reference_window_size or cfg.reference_window_size
            persist = cfg.trend_persistence or cfg.persistence
            cooldown = cfg.trend_cooldown or cfg.cooldown

            detector = DriftDetector(reference_window_size=ref_w)
            state = self._trend_state[metric]

            slope_arr = tdf["trend_slope"].to_numpy(dtype=float)
            n = len(tdf)
            for end in range(1, n + 1):
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1

                res = detector.detect(
                    slope_arr,
                    current_idx=end,
                    window_size=cur_w,
                    ks_threshold=cfg.trend_ks_threshold or cfg.ks_threshold,
                    psi_threshold=cfg.trend_psi_threshold or cfg.psi_threshold,
                    psi_bins=cfg.trend_psi_bins or cfg.psi_bins,
                    mde=None,
                    verbose=verbose,
                )
                if res is None:
                    continue

                drift = res["drift"]


                # optional effect-size gate on mean slope change
                if drift and cfg.trend_mde is not None:
                    ref_start, ref_end = res["reference_range"]
                    cur_start, cur_end = res["current_range"]
                    ref_mean = float(np.nanmean(slope_arr[ref_start:ref_end]))
                    cur_mean = float(np.nanmean(slope_arr[cur_start:cur_end]))
                    change = abs(cur_mean - ref_mean)
                    if  change <= cfg.trend_mde:

                        drift = False
                    elif verbose:
                        print(f"trend drift {metric} @ {end}: {drift}, change: {change}")

                # persistence on trend stream
                state["run"] = state["run"] + 1 if drift else 0

                if drift and state["run"] >= int(persist) and state["cooldown"] == 0:
                    ts = tdf.index[end - 1]
                    out_rows.append(
                        {
                            "timestamp": ts,
                            "idx": end - 1,
                            "metric": metric,
                            "drift": True,
                            "ks_p_value": res["ks"]["p_value"],
                            "psi": res["psi"]["value"],
                            "reference_range": res["reference_range"],
                            "current_range": res["current_range"],
                            "trend_rolling_window": tdf.attrs.get("trend_rolling_window"),
                            "trend_window_size": cur_w,
                            "trend_reference_window_size": ref_w,
                            "trend_source": tdf.attrs.get("trend_source"),
                            "trend_persistence": int(persist),
                            "trend_cooldown": int(cooldown),
                        }
                    )
                    state["cooldown"] = int(cooldown)
                    state["run"] = 0

        df_out = pd.DataFrame(out_rows)
        if not df_out.empty:
            df_out = df_out.sort_values(["timestamp", "metric"]).reset_index(drop=True)
        return df_out

    # -------------------------
    # Main alert stream
    # -------------------------
    def detect(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        feats = self.compute_features(df)
        all_alerts: List[Dict[str, Any]] = []

        for metric, feat_df in feats.items():
            cfg = self.metric_configs[metric]
            detector = self.detectors[metric]
            state = self._state[metric]

            feature_list = cfg.features if cfg.features is not None else self.DEFAULT_FEATURES
            min_k = max(1, int(cfg.min_features_to_alert))

            for end in range(1, len(feat_df) + 1):
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1

                drifted_features: List[str] = []
                evidence: Dict[str, Any] = {}

                for f in feature_list:
                    if f not in feat_df.columns:
                        continue
                    arr = feat_df[f].to_numpy(dtype=float)

                    # NOTE: check window_size passed as reference_window_size
                    res = detector.detect(
                        arr,
                        current_idx=end,
                        window_size=cfg.reference_window_size,
                        ks_threshold=cfg.ks_threshold,
                        psi_threshold=cfg.psi_threshold,
                        psi_bins=cfg.psi_bins,
                        mde=cfg.mde,
                        verbose=verbose,
                    )
                    if res is None:
                        continue

                    evidence[f] = {
                        "drift": res["drift"],
                        "ks_p_value": res["ks"]["p_value"],
                        "psi": res["psi"]["value"],
                    }
                    if res["drift"]:
                        drifted_features.append(f)


                drift_now = len(drifted_features) >= min_k
                state["run"] = state["run"] + 1 if drift_now else 0


                if drift_now and state["run"] >= cfg.persistence and state["cooldown"] == 0:
                    ts = feat_df.index[end - 1]
                    all_alerts.append(
                        {
                            "timestamp": ts,
                            "idx": end - 1,
                            "metric": metric,
                            "drift_features": drifted_features,
                            "num_drift_features": len(drifted_features),
                            "persistence": cfg.persistence,
                            "cooldown": cfg.cooldown,
                            "ks_threshold": cfg.ks_threshold,
                            "psi_threshold": cfg.psi_threshold,
                            "min_features_to_alert": min_k,
                            "evidence": evidence,
                            "status": "alert"
                        }
                    )
                    state["cooldown"] = cfg.cooldown

                elif drift_now and state["run"] >= cfg.persistence:
                    all_alerts.append(
                        {
                            "timestamp": ts,
                            "idx": end - 1,
                            "metric": metric,
                            "drift_features": drifted_features,
                            "num_drift_features": len(drifted_features),
                            "persistence": cfg.persistence,
                            "cooldown": cfg.cooldown,
                            "ks_threshold": cfg.ks_threshold,
                            "psi_threshold": cfg.psi_threshold,
                            "min_features_to_alert": min_k,
                            "evidence": evidence,
                            "status": "cooldown"
                        }
                    )
                    state["cooldown"] = cfg.cooldown

        alerts_df = pd.DataFrame(all_alerts)
        if not alerts_df.empty:
            alerts_df = alerts_df.sort_values(["timestamp", "metric"]).reset_index(drop=True)
        return alerts_df

    @property
    def features_(self) -> Dict[str, pd.DataFrame]:
        return self._features
    
    @property
    def trend_features_(self) -> Dict[str, pd.DataFrame]:
        return self._trend_features


# 4. Quality evaluation
def evaluate_alerts(true_drift_points, detected_points, tolerance=50):
    """
    true_drift_points: list[int] or list[pd.Timestamp]
    detected_points:   list[int] or list[pd.Timestamp]
    tolerance: allowed detection delay in points (if using int indices)
    Returns a dict with basic metrics.
    """
    if len(true_drift_points) == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "tp": 0, "fp": len(detected_points), "fn": 0}

    # If timestamps: map to ordinal positions is left to caller.
    # Here we assume int indices.
    true_set = set(true_drift_points)
    det = sorted(detected_points)

    matched_true = set()
    tp = 0
    for d in det:
        # match to nearest true point within tolerance
        candidates = [t for t in true_set if (t not in matched_true) and abs(d - t) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda t: abs(d - t))
            matched_true.add(best)
            tp += 1

    fp = len(det) - tp
    fn = len(true_set) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "fpr": fp/(len(det))}


# 5. Visualization
def plot_results(df, alerts_df, true_drifts):
    """

    df: DataFrame
    alerts_df: DataFrame with columns ['metric', 'idx'] (idx is the index in the series)
    true_drifts: dict[str, list[int]] with drift indices
    """
    metrics = ['requests', 'response_time', 'error_rate', 'cpu_usage']
    titles = [
        'Requests per Hour',
        'Response Time ms',
        'Error Rate %',
        'CPU Usage %'
    ]

    x = range(len(df))
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    for metric, title, ax in zip(metrics, titles, axes):
        ax.plot(x, df[metric].values, linewidth=1, alpha=0.7, label='Data')

        # true drifts
        first_true = True
        for drift_point in true_drifts.get(metric, []):
            ax.axvline(
                x=drift_point,
                color='red',
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label='True Drift' if first_true else ''
            )
            first_true = False

        # alerts
        if alerts_df is not None and len(alerts_df) > 0:
            am = alerts_df[(alerts_df["metric"] == metric) & (alerts_df["status"] == "alert")]
            first_alert = True
            for idx_point in am["idx"].tolist():
                ax.axvline(
                    x=int(idx_point),
                    color='orange',
                    linestyle='-',
                    linewidth=2,
                    alpha=0.7,
                    label='Alert' if first_alert else ''
                )
                first_alert = False

            am = alerts_df[(alerts_df["metric"] == metric) & (alerts_df["status"] == "cooldown")]
            first_alert = True
            for idx_point in am["idx"].tolist():
                ax.axvline(
                    x=int(idx_point),
                    color='lightgray',
                    linestyle='-',
                    linewidth=2,
                    alpha=0.2,
                    label='Cooldown' if first_alert else ''
                )
                first_alert = False

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_trend_alerts_all_metrics(
    df,
    alerts_trend,
    *,
    metrics=("requests", "response_time", "error_rate", "cpu_usage"),
    trend_features_by_metric=None,
    show_slope: bool = False,
    figsize=(15, 12),
):
    """
    Plots trend-drift alerts for all metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input data; contains columns listed in metrics.
    alerts_trend : pd.DataFrame
        DataFrame returned by detect_trend_drift with at least ['metric','idx'] columns.
    metrics : tuple/list
        Which metrics to plot.
    trend_features_by_metric : dict[str, pd.DataFrame] | None
        Dict {metric: trend_features_df} with a 'trend_slope' column.
    show_slope : bool
        Whether to draw slope charts (if trend_features_by_metric is provided).
    """
    # --- 1) Main plots: data + vertical alert lines ---
    x = range(len(df))
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

    if len(metrics) == 1:
        axes = [axes]

    for metric, ax in zip(metrics, axes):
        if metric not in df.columns:
            raise ValueError(f"df does not contain column '{metric}'")

        ax.plot(x, df[metric].astype(float).to_numpy(), linewidth=1, alpha=0.75, label="Data")

        am = None
        if alerts_trend is not None and len(alerts_trend) > 0:
            if "metric" not in alerts_trend.columns or "idx" not in alerts_trend.columns:
                raise ValueError("alerts_trend must contain columns ['metric','idx']")
            am = alerts_trend[alerts_trend["metric"] == metric]

        if am is not None and len(am) > 0:
            first = True
            for idx_point in am["idx"].tolist():
                ax.axvline(
                    x=int(idx_point),
                    color="orange",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.7,
                    label="Trend Drift Alert" if first else "",
                )
                first = False

        ax.set_title(f"{metric}: trend drift alerts", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Time (hours)")
    plt.tight_layout()
    plt.show()

    # --- 2) Optional: slope plots (each metric separately) ---
    if not show_slope or not trend_features_by_metric:
        return

    for metric in metrics:
        tf = trend_features_by_metric.get(metric)
        if tf is None or "trend_slope" not in tf.columns:
            continue

        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 4), sharex=True)
        slope = tf["trend_slope"].astype(float).to_numpy()
        ax2.plot(x, slope, linewidth=1, alpha=0.8, label="trend_slope")

        am = None
        if alerts_trend is not None and len(alerts_trend) > 0:
            am = alerts_trend[alerts_trend["metric"] == metric]

        if am is not None and len(am) > 0:
            first = True
            for idx_point in am["idx"].tolist():
                ax2.axvline(
                    x=int(idx_point),
                    color="orange",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.7,
                    label="Trend Drift Alert" if first else "",
                )
                first = False

        ax2.set_title(f"{metric}: trend_slope", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Slope (units/hour)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")
        ax2.set_xlabel("Time (hours)")
        plt.tight_layout()
        plt.show()
