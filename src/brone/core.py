from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "models" / "buffalo_sc" / "det_500m.onnx"
RECOGNIZER = ROOT / "vendor" / "edgeface" / "checkpoints" / "edgeface_xs_gamma_06.pt"
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Embedding invalid or zero")
    return x / n


def align(bgr, landmarks):
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Five finite landmarks required")
    transform, _ = cv2.estimateAffinePartial2D(points, TEMPLATE, method=cv2.LMEDS)
    if transform is None:
        raise ValueError("Alignment failed")
    return cv2.warpAffine(bgr, transform, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


class Detector:
    """SCRFD 500M KPS ONNX, CPU, matching the reference preprocessing/decoding."""

    def __init__(self, path=DETECTOR):
        import onnxruntime as ort
        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        if len(self.outputs) != 9:
            raise ValueError(f"Expected SCRFD KPS 9 outputs, got {len(self.outputs)}")

    def detect(self, bgr, score_threshold=0.5, nms_threshold=0.4):
        h, w = bgr.shape[:2]
        scale = min(640 / w, 640 / h)
        resized = cv2.resize(bgr, (round(w * scale), round(h * scale)))
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (640, 640), (127.5,) * 3, swapRB=True)
        outs = self.session.run(self.outputs, {self.input: blob})
        candidates = []
        for i, stride in enumerate((8, 16, 32)):
            scores = outs[i].reshape(-1)
            bbox = outs[i + 3].reshape(-1, 4) * stride
            kps = outs[i + 6].reshape(-1, 10) * stride
            yy, xx = np.mgrid[:640 // stride, :640 // stride]
            centers = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centers = np.repeat(centers, 2, axis=0)
            for j in np.flatnonzero(scores >= score_threshold):
                c = centers[j]
                box = (np.array([c[0] - bbox[j, 0], c[1] - bbox[j, 1],
                                 c[0] + bbox[j, 2], c[1] + bbox[j, 3]]) / scale)
                points = (kps[j].reshape(5, 2) + c) / scale
                candidates.append((float(scores[j]), box, points))
        candidates.sort(key=lambda x: x[0], reverse=True)
        kept = []
        for item in candidates:
            a = item[1]
            if all(_iou(a, old[1]) <= nms_threshold for old in kept):
                kept.append(item)
        return kept


def _iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


class Embedder:
    def __init__(self, path=RECOGNIZER):
        import torch
        sys.path.insert(0, str(ROOT / "vendor" / "edgeface"))
        from backbones import get_model
        self.path = Path(path)
        self.model = get_model("edgeface_xs_gamma_06")
        self.model.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True), strict=True)
        self.model.eval()
        self.torch = torch

    def embed(self, aligned_bgr):
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        x = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        x = (x - 0.5) / 0.5
        with self.torch.inference_mode():
            y = self.model(x).cpu().numpy()[0]
        if y.size != 512:
            raise ValueError(f"Expected 512-D embedding, got {y.size}")
        return unit(y)


class Pipeline:
    def __init__(self):
        start = time.perf_counter()
        self.detector = Detector()
        self.embedder = Embedder()
        self.load_ms = (time.perf_counter() - start) * 1000
        self.versions = {"detector_sha256": sha256(DETECTOR), "recognizer_sha256": sha256(RECOGNIZER)}

    def process(self, path, debug_dir=None):
        start = time.perf_counter()
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        t = time.perf_counter()
        faces = self.detector.detect(bgr)
        det_ms = (time.perf_counter() - t) * 1000
        if len(faces) != 1:
            raise ValueError(f"Expected exactly one face, found {len(faces)}: {path}")
        score, box, points = faces[0]
        t = time.perf_counter()
        crop = align(bgr, points)
        align_ms = (time.perf_counter() - t) * 1000
        if debug_dir:
            out = Path(debug_dir)
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / (Path(path).stem + "_aligned.png")), crop)
        t = time.perf_counter()
        vector = self.embedder.embed(crop)
        embed_ms = (time.perf_counter() - t) * 1000
        return vector, {"detector_score": score, "bbox": box.tolist(),
                        "landmarks": points.tolist(), "detection_ms": det_ms,
                        "alignment_ms": align_ms, "embedding_ms": embed_ms,
                        "latency_ms": (time.perf_counter() - start) * 1000}


def match(vector, gallery, threshold):
    if not gallery:
        raise ValueError("Gallery is empty")
    scores = {identity: float(np.dot(unit(vector), unit(reference)))
              for identity, reference in gallery.items()}
    identity = max(scores, key=scores.get)
    score = scores[identity]
    return {"predicted_id": identity if score >= threshold else "unknown",
            "nearest_id": identity, "score": score, "accepted": score >= threshold}


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"path", "subject_alias", "split", "device", "session", "consent"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest needs columns: {sorted(required)}")
    if any(r["consent"].lower() not in ("yes", "true", "1") for r in rows):
        raise ValueError("Every image needs recorded consent")
    if any(r["split"] not in ("S1", "S2", "S3") for r in rows):
        raise ValueError("Split must be S1, S2, or S3")
    resolved = []
    seen = set()
    for r in rows:
        p = (ROOT / "data" / r["path"]).resolve()
        if not p.is_relative_to(ROOT / "data"):
            raise ValueError("Image path must stay inside data/")
        if p in seen:
            raise ValueError(f"Image reused across manifest rows: {p}")
        seen.add(p)
        r["absolute_path"] = str(p)
        resolved.append(r)
    return resolved


def save_gallery(path, gallery, versions):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"model": versions, "embeddings":
                         {k: v.tolist() for k, v in gallery.items()}}, indent=2), encoding="utf-8")


def load_gallery(path, versions):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["model"] != versions:
        raise ValueError("Gallery model hashes do not match current models")
    return {k: unit(v) for k, v in data["embeddings"].items()}
