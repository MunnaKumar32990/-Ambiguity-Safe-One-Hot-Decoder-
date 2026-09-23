"""
app.py
======
Flask REST API for the Ambiguity-Safe Inverse Decoding system.

Exposes the core research prototype (src/) as HTTP endpoints for the
interactive web frontend. No logic is duplicated — all computations
delegate to the existing src/ modules.

Endpoints:
    POST /api/analyze   — One-shot: fit → encode → safe decode
    GET  /api/presets    — List available preset experiments
    GET  /api/preset/<k> — Get a specific preset's data
    GET  /api/health     — Health check

Run:
    python backend/app.py
    or
    python run_app.py  (recommended — also opens frontend)
"""

import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.safe_decoder import AmbiguitySafeOneHotDecoder
from src.ambiguity_detector import AmbiguityStatus
from src.evaluator import ExperimentEvaluator
from backend.presets import get_preset_names, get_preset, PRESETS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
CORS(app)

# Frontend directory for serving static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

evaluator = ExperimentEvaluator()


# ---------------------------------------------------------------------------
# Serve frontend files
# ---------------------------------------------------------------------------
@app.route("/")
def serve_index():
    """Serve the main frontend page."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/css/<path:filename>")
def serve_css(filename):
    """Serve CSS files."""
    return send_from_directory(os.path.join(FRONTEND_DIR, "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    """Serve JavaScript files."""
    return send_from_directory(os.path.join(FRONTEND_DIR, "js"), filename)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "project": "Ambiguity-Safe Inverse Decoding",
        "version": "1.0.0",
    })


@app.route("/api/presets", methods=["GET"])
def list_presets():
    """Return all available preset experiment names and descriptions."""
    return jsonify({"presets": get_preset_names()})


@app.route("/api/preset/<key>", methods=["GET"])
def get_preset_data(key):
    """Return full data for a specific preset."""
    preset = get_preset(key)
    if preset is None:
        return jsonify({"error": f"Preset '{key}' not found"}), 404
    return jsonify({"preset": preset})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    One-shot analysis: fit encoder → encode test data → safe decode.

    Request JSON:
    {
        "train_data": [["Female"], ["Male"]],
        "test_data": [["Female"], ["Male"], ["Unknown"]],
        "ground_truth": [["Female"], ["Male"], ["Unknown"]],  // optional
        "feature_names": ["Gender"],  // optional
        "drop": "if_binary",
        "handle_unknown": "ignore"
    }

    Response JSON:
    {
        "success": true,
        "config": { ... },
        "metadata": { ... },
        "results": [ ... ],
        "metrics": { ... },
        "summary": { ... }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Extract parameters
        train_data = data.get("train_data")
        test_data = data.get("test_data")
        ground_truth = data.get("ground_truth")
        feature_names = data.get("feature_names")
        drop = data.get("drop")  # Can be None, "first", "if_binary"
        handle_unknown = data.get("handle_unknown", "ignore")

        if not train_data or not test_data:
            return jsonify({"error": "train_data and test_data are required"}), 400

        # Handle "null" string from frontend
        if drop == "null" or drop == "":
            drop = None

        # ---------------------------------------------------------------
        # Step 1: Fit and transform using the safe decoder
        # ---------------------------------------------------------------
        decoder = AmbiguitySafeOneHotDecoder(
            drop=drop,
            handle_unknown=handle_unknown,
        )
        decoder.fit(train_data)
        X_encoded = decoder.transform(test_data)
        safe_results = decoder.safe_inverse_transform(X_encoded)

        # Also get the baseline sklearn results
        baseline_decoded = decoder.baseline_inverse_transform(X_encoded)

        # ---------------------------------------------------------------
        # Step 2: Build metadata response
        # ---------------------------------------------------------------
        meta = decoder.metadata
        metadata_response = {
            "n_features": meta.n_features,
            "drop_config": meta.drop_config,
            "handle_unknown": meta.handle_unknown,
            "total_encoded_columns": meta.total_encoded_columns,
            "features": [],
        }
        for fm in meta.features:
            metadata_response["features"].append({
                "feature_index": fm.feature_index,
                "feature_name": (
                    feature_names[fm.feature_index]
                    if feature_names and fm.feature_index < len(feature_names)
                    else f"Feature_{fm.feature_index}"
                ),
                "categories": [str(c) for c in fm.categories],
                "drop_config": fm.drop_config,
                "dropped_category": str(fm.dropped_category) if fm.dropped_category else None,
                "handle_unknown": fm.handle_unknown,
                "is_binary": fm.is_binary,
                "can_produce_all_zeros": fm.can_produce_all_zeros(),
                "n_encoded_columns": fm.n_encoded_columns,
                "encoded_col_range": [fm.encoded_col_start, fm.encoded_col_end],
            })

        # ---------------------------------------------------------------
        # Step 3: Build per-sample results
        # ---------------------------------------------------------------
        results_response = []
        for i, r in enumerate(safe_results):
            input_val = test_data[i] if i < len(test_data) else None
            gt_val = ground_truth[i] if ground_truth and i < len(ground_truth) else None
            baseline_val = [str(v) if v is not None else None for v in baseline_decoded[i]]

            # Per-feature detail
            feature_details = []
            for j, rep in enumerate(r.feature_reports):
                fname = (
                    feature_names[j]
                    if feature_names and j < len(feature_names)
                    else f"Feature_{j}"
                )
                feature_details.append({
                    "feature_name": fname,
                    "feature_index": rep.feature_index,
                    "encoded_sub_vector": rep.encoded_sub_vector,
                    "status": rep.status.value,
                    "possible_values": [str(v) for v in rep.possible_values],
                    "sklearn_decoded": str(rep.sklearn_decoded) if rep.sklearn_decoded is not None else None,
                    "reason": rep.reason,
                })

            # Baseline correctness
            baseline_correct = None
            if gt_val is not None:
                baseline_correct = baseline_val == [str(g) for g in gt_val]

            results_response.append({
                "sample_index": i,
                "input": [str(v) for v in input_val] if input_val else None,
                "ground_truth": [str(v) for v in gt_val] if gt_val else None,
                "encoded": [int(v) for v in r.encoded],
                "baseline_decoded": baseline_val,
                "baseline_correct": baseline_correct,
                "row_status": r.row_status.value,
                "safe_output": [str(v) if v is not None else None for v in r.safe_output],
                "possible_values": [[str(v) for v in pv] for pv in r.possible_values],
                "confidence": r.confidence,
                "reasons": r.reasons,
                "feature_details": feature_details,
            })

        # ---------------------------------------------------------------
        # Step 4: Compute metrics
        # ---------------------------------------------------------------
        metrics = evaluator.compute_metrics(
            experiment_name="web_analysis",
            results=safe_results,
            ground_truth=ground_truth,
            notes="Interactive web analysis",
        )
        metrics_response = {
            "total_samples": metrics.total_samples,
            "safe_count": metrics.safe_count,
            "ambiguous_count": metrics.ambiguous_count,
            "unknown_count": metrics.unknown_count,
            "baseline_correct": metrics.baseline_correct,
            "baseline_incorrect": metrics.baseline_incorrect,
            "detection_rate": round(metrics.detection_rate * 100, 1),
            "safe_handling_rate": round(metrics.safe_handling_rate * 100, 1),
        }

        # ---------------------------------------------------------------
        # Step 5: Build summary
        # ---------------------------------------------------------------
        summary = {
            "total_samples": metrics.total_samples,
            "baseline_accuracy": (
                f"{metrics.baseline_correct}/{metrics.total_samples} "
                f"({metrics.baseline_correct / metrics.total_samples * 100:.0f}%)"
                if metrics.total_samples > 0 else "N/A"
            ),
            "silent_errors": metrics.baseline_incorrect,
            "ambiguities_detected": metrics.ambiguous_count,
            "detection_rate": f"{metrics.detection_rate * 100:.0f}%",
            "safe_handling_rate": f"{metrics.safe_handling_rate * 100:.0f}%",
            "key_finding": _generate_key_finding(metrics, drop),
        }

        return jsonify({
            "success": True,
            "config": {
                "drop": drop,
                "handle_unknown": handle_unknown,
                "n_train_samples": len(train_data),
                "n_test_samples": len(test_data),
            },
            "metadata": metadata_response,
            "results": results_response,
            "metrics": metrics_response,
            "summary": summary,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }), 500


def _generate_key_finding(metrics, drop):
    """Generate a human-readable key finding from metrics."""
    if metrics.baseline_incorrect > 0 and metrics.ambiguous_count > 0:
        return (
            f"sklearn silently misidentified {metrics.baseline_incorrect} sample(s). "
            f"Our system detected {metrics.ambiguous_count} ambiguous case(s) and "
            f"withheld the reconstruction, achieving a {metrics.detection_rate * 100:.0f}% "
            f"detection rate with zero silent errors."
        )
    elif metrics.unknown_count > 0 and metrics.ambiguous_count == 0:
        return (
            f"With drop={drop!r}, no ambiguity collision occurs. "
            f"{metrics.unknown_count} unseen category(s) were correctly classified "
            f"as UNKNOWN rather than AMBIGUOUS."
        )
    else:
        return (
            f"All {metrics.total_samples} samples were decoded safely. "
            f"No ambiguity detected under this configuration."
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Ambiguity-Safe Decoder — API Server")
    print("=" * 60)
    print(f"  Frontend: http://localhost:5000")
    print(f"  API Base: http://localhost:5000/api")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
