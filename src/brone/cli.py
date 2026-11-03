import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import (DETECTOR, RECOGNIZER, ROOT, Pipeline, load_gallery, match,
                   read_manifest, save_gallery, sha256)


def main():
    parser = argparse.ArgumentParser(prog="brone", description="BRONE E0 local face recognition")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    enroll = sub.add_parser("enroll")
    enroll.add_argument("--manifest", default="data/manifest.csv")
    enroll.add_argument("--gallery", default="features/edgeface_xs/gallery.json")
    enroll.add_argument("--debug-dir", default="data/aligned/enroll")
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--manifest", default="data/manifest.csv")
    calibrate.add_argument("--gallery", default="features/edgeface_xs/gallery.json")
    calibrate.add_argument("--out", default="configs/threshold.json")
    infer = sub.add_parser("infer")
    infer.add_argument("image")
    infer.add_argument("--gallery", default="features/edgeface_xs/gallery.json")
    infer.add_argument("--threshold", default="configs/threshold.json")
    infer.add_argument("--debug-dir", default="data/aligned/infer")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", default="data/manifest.csv")
    evaluate.add_argument("--gallery", default="features/edgeface_xs/gallery.json")
    evaluate.add_argument("--threshold", default="configs/threshold.json")
    evaluate.add_argument("--out", default="results/e0/predictions.csv")
    args = parser.parse_args()
    if args.command == "check":
        for name, path in (("SCRFD_500M_KPS", DETECTOR), ("EdgeFace-XS", RECOGNIZER)):
            print(json.dumps({"model": name, "exists": path.is_file(),
                              "sha256": sha256(path) if path.is_file() else None,
                              "path": str(path)}))
        pipeline = Pipeline()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        embedding = pipeline.embedder.embed(dummy)
        print(json.dumps({"status": "models_loaded", "embedding_dim": len(embedding),
                          "embedding_norm": float(np.linalg.norm(embedding)),
                          "model_load_ms": pipeline.load_ms}))
        return
    pipeline = Pipeline()
    rows = read_manifest(args.manifest) if args.command != "infer" else None
    if args.command == "enroll":
        selected = [r for r in rows if r["split"] == "S1" and r["subject_alias"] != "unknown"]
        if len(selected) != len({r["subject_alias"] for r in selected}):
            raise ValueError("E0 requires one S1 enrollment image per identity")
        if not selected:
            raise ValueError("No S1 enrollment images")
        gallery = {}
        for row in selected:
            gallery[row["subject_alias"]], _ = pipeline.process(row["absolute_path"], args.debug_dir)
        save_gallery(args.gallery, gallery, pipeline.versions)
        print(json.dumps({"enrolled": sorted(gallery), "gallery": args.gallery}))
        return
    gallery = load_gallery(args.gallery, pipeline.versions)
    if args.command == "calibrate":
        validation = [r for r in rows if r["split"] == "S2"]
        if not validation or not any(r["subject_alias"] == "unknown" for r in validation):
            raise ValueError("S2 needs known and unknown probes; no threshold will be invented")
        scored = []
        for row in validation:
            vec, _ = pipeline.process(row["absolute_path"])
            scored.append((row["subject_alias"], match(vec, gallery, -1)["score"]))
        # Conservative empirical zero-false-accept threshold on S2.
        unknown_max = max(s for label, s in scored if label == "unknown")
        threshold = float(np.nextafter(np.float32(unknown_max), np.float32(1)))
        if threshold > 1:
            raise ValueError("Cannot calibrate threshold")
        payload = {"threshold": threshold, "method": "above_max_unknown_S2",
                   "validation_samples": len(scored), "model": pipeline.versions}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return
    threshold_data = json.loads(Path(args.threshold).read_text(encoding="utf-8"))
    if threshold_data["model"] != pipeline.versions:
        raise ValueError("Threshold model hashes do not match")
    threshold = threshold_data["threshold"]
    if args.command == "infer":
        vec, meta = pipeline.process(args.image, args.debug_dir)
        print(json.dumps({**match(vec, gallery, threshold), **meta,
                          "model": pipeline.versions, "model_load_ms": pipeline.load_ms}))
        return
    test = [r for r in rows if r["split"] == "S3"]
    if not test:
        raise ValueError("No S3 test images")
    output = []
    for row in test:
        try:
            vec, meta = pipeline.process(row["absolute_path"], "data/aligned/test")
            decision = match(vec, gallery, threshold)
            output.append({"path": row["path"], "ground_truth": row["subject_alias"],
                           "device": row["device"], "session": row["session"],
                           "status": "ok", **decision, **meta,
                           "model_version": pipeline.versions["recognizer_sha256"],
                           "threshold": threshold})
        except ValueError as exc:
            output.append({"path": row["path"], "ground_truth": row["subject_alias"],
                           "device": row["device"], "session": row["session"],
                           "status": str(exc), "predicted_id": "no_face"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(r.keys() for r in output)))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(output)
    ok = [r for r in output if r["status"] == "ok"]
    known = [r for r in ok if r["ground_truth"] != "unknown"]
    unknown = [r for r in ok if r["ground_truth"] == "unknown"]
    summary = {"samples": len(output), "detection_success_rate": len(ok) / len(output),
               "known_identification_accuracy": (sum(r["predicted_id"] == r["ground_truth"] for r in known) / len(known)) if known else None,
               "false_accept_rate_unknown": (sum(r["accepted"] for r in unknown) / len(unknown)) if unknown else None,
               "false_reject_rate_known": (sum(not r["accepted"] for r in known) / len(known)) if known else None,
               "end_to_end_latency_ms_p50": float(np.percentile([r["latency_ms"] for r in ok], 50)) if ok else None,
               "end_to_end_latency_ms_p95": float(np.percentile([r["latency_ms"] for r in ok], 95)) if ok else None,
               "model_load_ms": pipeline.load_ms, "threshold": threshold,
               "model": pipeline.versions}
    out.with_name("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(output), "predictions_csv": str(out),
                      "summary_json": str(out.with_name("summary.json"))}))


if __name__ == "__main__":
    main()
