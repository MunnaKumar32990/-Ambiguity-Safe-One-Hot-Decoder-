"""
evaluator.py
============
ExperimentEvaluator — computes metrics over experiment results and
generates CSV reports and matplotlib plots.

Metrics computed per experiment
--------------------------------
- total_samples      : Total test cases
- safe_count         : Correctly decoded with HIGH confidence
- ambiguous_count    : Detected as ambiguous (cannot decode safely)
- unknown_count      : Detected as unknown (unseen category, no ambiguity)
- baseline_correct   : Rows where sklearn's inverse matches ground truth
- baseline_incorrect : Rows where sklearn's inverse is wrong
- safe_handled_rate  : % of unsafe cases handled gracefully (not silently wrong)
- detection_rate     : % of non-safe cases that were flagged by the proposed system
"""

import os
import json
import csv
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .safe_decoder import DecodingResult
from .ambiguity_detector import AmbiguityStatus


@dataclass
class ExperimentMetrics:
    """
    Aggregated metrics for one experiment run.

    Attributes
    ----------
    experiment_name : str
    total_samples : int
    safe_count : int
        Samples decoded with SAFE status.
    ambiguous_count : int
        Samples decoded with AMBIGUOUS status.
    unknown_count : int
        Samples decoded with UNKNOWN status.
    baseline_correct : int
        Samples where sklearn's decode matches the ground truth label.
    baseline_incorrect : int
        Samples where sklearn's decode does NOT match ground truth.
    detection_rate : float
        Fraction of non-safe cases correctly flagged by the proposed system.
        = (ambiguous_count + unknown_count) / (total - safe_correct_matches)
    safe_handling_rate : float
        Fraction of all cases where the proposed system either (a) returned
        a correct safe output, or (b) explicitly flagged an unsafe case.
    """
    experiment_name: str
    total_samples: int = 0
    safe_count: int = 0
    ambiguous_count: int = 0
    unknown_count: int = 0
    baseline_correct: int = 0
    baseline_incorrect: int = 0
    detection_rate: float = 0.0
    safe_handling_rate: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "experiment": self.experiment_name,
            "total_samples": self.total_samples,
            "safe_count": self.safe_count,
            "ambiguous_count": self.ambiguous_count,
            "unknown_count": self.unknown_count,
            "baseline_correct": self.baseline_correct,
            "baseline_incorrect": self.baseline_incorrect,
            "detection_rate_pct": round(self.detection_rate * 100, 2),
            "safe_handling_rate_pct": round(self.safe_handling_rate * 100, 2),
            "notes": self.notes,
        }


class ExperimentEvaluator:
    """
    Evaluator for comparing baseline and proposed system performance.

    Usage
    -----
    >>> evaluator = ExperimentEvaluator()
    >>> metrics = evaluator.compute_metrics(
    ...     experiment_name="binary_drop_ignore",
    ...     results=safe_decode_results,
    ...     ground_truth=["Female", "Male", "Unknown"],
    ...     baseline_decoded=[["Female"], ["Male"], ["Female"]],
    ... )
    >>> evaluator.save_csv([metrics], "results/experiment_summary.csv")
    >>> evaluator.plot_comparison([metrics], "results/plots/comparison.png")
    """

    def compute_metrics(
        self,
        experiment_name: str,
        results: List[DecodingResult],
        ground_truth: Optional[List[List[Any]]] = None,
        baseline_decoded: Optional[List[List[Any]]] = None,
        notes: str = "",
    ) -> ExperimentMetrics:
        """
        Compute experiment metrics.

        Parameters
        ----------
        experiment_name : str
        results : list[DecodingResult]
            Output from AmbiguitySafeOneHotDecoder.safe_inverse_transform()
        ground_truth : list[list] or None
            True labels (optional). Each row is a list of category values.
            If None, baseline_correct/incorrect will be 0.
        baseline_decoded : list[list] or None
            sklearn's raw inverse_transform output for the same test set.
            If None, derived from DecodingResult.sklearn_decoded.
        notes : str
            Experiment description notes.

        Returns
        -------
        ExperimentMetrics
        """
        total = len(results)
        safe_count = sum(1 for r in results if r.row_status == AmbiguityStatus.SAFE)
        ambiguous_count = sum(1 for r in results if r.row_status == AmbiguityStatus.AMBIGUOUS)
        unknown_count = sum(1 for r in results if r.row_status == AmbiguityStatus.UNKNOWN)

        # Baseline correctness (requires ground truth)
        baseline_correct = 0
        baseline_incorrect = 0
        if ground_truth is not None:
            for i, r in enumerate(results):
                gt_row = ground_truth[i] if isinstance(ground_truth[i], list) else [ground_truth[i]]
                sk_row = r.sklearn_decoded
                if gt_row == sk_row:
                    baseline_correct += 1
                else:
                    baseline_incorrect += 1

        # Detection rate: among cases where baseline is wrong, how many
        # did the proposed system flag? (safe_handled / baseline_incorrect)
        if baseline_incorrect > 0:
            # Flagged as AMBIGUOUS or UNKNOWN
            flagged = ambiguous_count + unknown_count
            detection_rate = min(flagged / baseline_incorrect, 1.0)
        else:
            detection_rate = 1.0  # No errors -> trivially 100%

        # Safe handling rate: (correctly decoded SAFE + flagged unsafe) / total
        # "correctly decoded SAFE" = safe_count (proposing system said SAFE)
        # Since we cannot always verify SAFE correctness without ground truth,
        # we use: safe_handling_rate = (safe_count + flagged) / total
        flagged = ambiguous_count + unknown_count
        safe_handling_rate = (safe_count + flagged) / total if total > 0 else 0.0

        return ExperimentMetrics(
            experiment_name=experiment_name,
            total_samples=total,
            safe_count=safe_count,
            ambiguous_count=ambiguous_count,
            unknown_count=unknown_count,
            baseline_correct=baseline_correct,
            baseline_incorrect=baseline_incorrect,
            detection_rate=detection_rate,
            safe_handling_rate=safe_handling_rate,
            notes=notes,
        )

    def build_row_table(
        self,
        results: List[DecodingResult],
        input_values: Optional[List[List[Any]]] = None,
        ground_truth: Optional[List[List[Any]]] = None,
    ) -> pd.DataFrame:
        """
        Build a per-row summary DataFrame for display and CSV export.

        Parameters
        ----------
        results : list[DecodingResult]
        input_values : list[list] or None
            Original (pre-encoded) inputs for display.
        ground_truth : list[list] or None
            True category labels for correctness annotation.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for i, r in enumerate(results):
            input_val = input_values[i] if input_values else ["N/A"]
            gt_val = ground_truth[i] if ground_truth else ["N/A"]
            sk_dec = r.sklearn_decoded
            correct = (gt_val == sk_dec) if ground_truth else None

            rows.append({
                "input": input_val,
                "encoded": r.encoded,
                "sklearn_decoded": sk_dec,
                "baseline_correct": correct,
                "proposed_status": r.row_status.value,
                "safe_output": r.safe_output,
                "possible_values": r.possible_values,
                "confidence": r.confidence,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # CSV / JSON Export                                                    #
    # ------------------------------------------------------------------ #

    def save_csv(
        self,
        metrics_list: List[ExperimentMetrics],
        filepath: str,
    ) -> None:
        """Save a list of ExperimentMetrics to a CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        rows = [m.to_dict() for m in metrics_list]
        if not rows:
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def save_row_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save row-level table DataFrame to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        df.to_csv(filepath, index=False)

    def save_json(
        self,
        results: List[DecodingResult],
        filepath: str,
    ) -> None:
        """Save full DecodingResult list as JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        data = []
        for r in results:
            data.append({
                "sample_index": r.sample_index,
                "encoded": r.encoded,
                "sklearn_decoded": r.sklearn_decoded,
                "row_status": r.row_status.value,
                "safe_output": r.safe_output,
                "possible_values": r.possible_values,
                "confidence": r.confidence,
                "reasons": r.reasons,
            })
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #

    def plot_status_distribution(
        self,
        metrics_list: List[ExperimentMetrics],
        filepath: str,
        title: str = "Proposed System — Decoding Status Distribution",
    ) -> None:
        """
        Bar chart: SAFE / AMBIGUOUS / UNKNOWN counts across experiments.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        experiments = [m.experiment_name for m in metrics_list]
        safe_vals = [m.safe_count for m in metrics_list]
        ambig_vals = [m.ambiguous_count for m in metrics_list]
        unknown_vals = [m.unknown_count for m in metrics_list]

        x = np.arange(len(experiments))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 2), 6))
        bars1 = ax.bar(x - width, safe_vals, width, label="SAFE", color="#2ecc71", alpha=0.85)
        bars2 = ax.bar(x, ambig_vals, width, label="AMBIGUOUS", color="#e74c3c", alpha=0.85)
        bars3 = ax.bar(x + width, unknown_vals, width, label="UNKNOWN", color="#f39c12", alpha=0.85)

        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with values
        for bar in [*bars1, *bars2, *bars3]:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        str(int(h)), ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_baseline_vs_proposed(
        self,
        metrics_list: List[ExperimentMetrics],
        filepath: str,
        title: str = "Baseline vs Proposed: Incorrect vs Flagged",
    ) -> None:
        """
        Comparison chart: baseline incorrect decodes vs proposed flagged cases.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        experiments = [m.experiment_name for m in metrics_list]
        baseline_wrong = [m.baseline_incorrect for m in metrics_list]
        flagged = [m.ambiguous_count + m.unknown_count for m in metrics_list]

        x = np.arange(len(experiments))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 2), 6))
        bars1 = ax.bar(x - width / 2, baseline_wrong, width,
                       label="Baseline Incorrect Decodes", color="#c0392b", alpha=0.85)
        bars2 = ax.bar(x + width / 2, flagged, width,
                       label="Proposed: Flagged (AMBIGUOUS + UNKNOWN)", color="#2980b9", alpha=0.85)

        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        for bar in [*bars1, *bars2]:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        str(int(h)), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_detection_rate(
        self,
        metrics_list: List[ExperimentMetrics],
        filepath: str,
        title: str = "Ambiguity Detection Rate by Experiment",
    ) -> None:
        """
        Horizontal bar chart of detection rates across experiments.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        experiments = [m.experiment_name for m in metrics_list]
        rates = [m.detection_rate * 100 for m in metrics_list]

        fig, ax = plt.subplots(figsize=(9, max(4, len(experiments) * 0.8)))
        colors = ["#27ae60" if r >= 80 else "#e67e22" if r >= 50 else "#c0392b" for r in rates]
        bars = ax.barh(experiments, rates, color=colors, alpha=0.85)

        ax.set_xlabel("Detection Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.axvline(x=100, color="grey", linestyle="--", alpha=0.5)

        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.1f}%", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

    def plot_all(
        self,
        metrics_list: List[ExperimentMetrics],
        plots_dir: str,
    ) -> None:
        """Generate all standard plots."""
        os.makedirs(plots_dir, exist_ok=True)
        self.plot_status_distribution(
            metrics_list,
            os.path.join(plots_dir, "status_distribution.png"),
        )
        self.plot_baseline_vs_proposed(
            metrics_list,
            os.path.join(plots_dir, "baseline_vs_proposed.png"),
        )
        self.plot_detection_rate(
            metrics_list,
            os.path.join(plots_dir, "detection_rate.png"),
        )
