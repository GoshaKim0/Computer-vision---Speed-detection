# -------------------------------------------------------------
# Traffic speed detection/tracking with per-direction counting.
# YOLO + ByteTrack, BEV homography per ROI, median speed over a short window.
# Quick setup: --setup N → for each ROI: 4 clicks (BL→BR→TR→TL) + 1 line click.
# Run:
#   python traffic_speed_limit_v3.py --video ".\\traffic.mp4" --setup 2 --config ".\\config.yaml"
#   python traffic_speed_limit_v3.py --video ".\\traffic.mp4" --config ".\\config.yaml"
# -------------------------------------------------------------

from __future__ import annotations
import os, sys, math, yaml
from typing import List, Tuple, Dict, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np
import cv2
from ultralytics import YOLO

# ---------- DEFAULTS (overridable via config.yaml) ----------
DEFAULT_CFG = {
    "speed_limit_kmh": 130,
    "classes": [2, 3, 5, 7],  # car, motorcycle, bus, truck
    "speed_smoothing_sec": 0.8,
    "min_track_frames": 5,
    "conf": 0.30,
    "iou": 0.5,
    "output_path": "out/output.mp4",
    # Two ROIs by default. width_m = 3.6 m * number of lanes covered by the trapezoid.
    "rois": [
        {
            "name": "right",
            "src": [[900, 640], [1210, 640], [1085, 420], [885, 420]],  # BL, BR, TR, TL
            "width_m": 14.4,  # 4 lanes = 4*3.6
            "length_m": 100.0,  # along the road
            "count_line_y_m": 35.0,
            "flow_direction": "forward",  # away from camera
        },
        {
            "name": "left",
            "src": [[380, 640], [120, 640], [240, 420], [460, 420]],
            "width_m": 14.4,
            "length_m": 100.0,
            "count_line_y_m": 35.0,
            "flow_direction": "backward",  # towards camera
        },
    ],
}

COCO_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ---------- Utils ----------
def load_config(path: str | None) -> dict:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = DEFAULT_CFG.copy()
        if "rois" in data:
            cfg["rois"] = data["rois"]
            for k, v in data.items():
                if k != "rois":
                    cfg[k] = v
        else:
            cfg.update(data)
        return cfg
    return DEFAULT_CFG.copy()


def save_config(path: str, cfg: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def homography_px_to_meters(
    src_pts_px: np.ndarray, width_m: float, length_m: float
) -> np.ndarray:
    """
    H matrix: pixels (ROI) -> meters (BEV).
    Target rectangle: (0,0)-(width,0)-(width,length)-(0,length).
    """
    src = src_pts_px.astype(np.float32)
    dst = np.array(
        [[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(src, dst)


def persp_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def draw_label(img, text: str, x: int, y: int):
    # tiny label helper (black bg + white text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 0.55, 2
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    cv2.rectangle(
        img, (x, y - tht - 6), (x + tw + 6, y), (0, 0, 0), -1, lineType=cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (x + 3, y - 3),
        font,
        scale,
        (255, 255, 255),
        th,
        lineType=cv2.LINE_AA,
    )


def robust_speed_kmh(hist: List[Tuple[int, float, float]], fps: float) -> float:
    """
    Median of instantaneous speeds (m→km/h) between consecutive points.
    Clamp outliers >220 km/h.
    """
    if len(hist) < 2:
        return 0.0
    vals = []
    for (f0, x0, y0), (f1, x1, y1) in zip(hist[:-1], hist[1:]):
        dt = max(1e-6, (f1 - f0) / fps)
        sp = (math.hypot(x1 - x0, y1 - y0) / dt) * 3.6
        if 0.0 <= sp <= 220.0:
            vals.append(sp)
    return float(np.median(vals)) if vals else 0.0


def crossed(prev_y: float, now_y: float, line_y: float, direction: str) -> bool:
    # simple one-line crossing check respecting flow direction
    up = prev_y <= line_y < now_y
    down = prev_y >= line_y > now_y
    if direction == "forward":
        return up
    if direction == "backward":
        return down
    return up or down


def bbox_hits_poly(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cx: float,
    cy: float,
    poly: np.ndarray,
    pad: int = 4,
) -> bool:
    """
    Inside ROI if center OR any bbox corner is inside polygon.
    Small pad reduces misses at ROI edges.
    """
    x1p, y1p, x2p, y2p = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    pts = [(cx, cy), (x1p, y1p), (x2p, y1p), (x2p, y2p), (x1p, y2p)]
    for px, py in pts:
        if cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0:
            return True
    return False


# ---------- SETUP (mouse annotation) ----------
@dataclass
class ROIState:
    src: List[Tuple[int, int]] = field(default_factory=list)
    line_pt: Tuple[int, int] | None = None
    name: str = ""


def setup_interactive(video_path: str, cfg_path: str, n_rois: int):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[ERROR] Failed to open video.")
            sys.exit(1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[ERROR] Failed to read frame.")
        sys.exit(1)

    states: List[ROIState] = [ROIState(name=f"dir{i+1}") for i in range(n_rois)]
    idx = 0
    WIN = "SETUP"  # fixed name → stable mouse callback binding

    def on_mouse(event, x, y, flags, param):
        nonlocal idx
        st = states[idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(st.src) < 4:
                st.src.append((x, y))
                print(f"[CLICK] ROI {idx+1}: pt{len(st.src)} = {(x,y)}")
            elif st.line_pt is None:
                st.line_pt = (x, y)
                print(f"[CLICK] ROI {idx+1}: count line y-px = {y}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            st.src.clear()
            st.line_pt = None
            print(f"[RESET] ROI {idx+1}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    while True:
        cv2.setMouseCallback(WIN, on_mouse)  # re-bind each loop for safety
        vis = frame.copy()
        st = states[idx]

        draw_label(
            vis,
            f"ROI {idx+1}/{n_rois}: LMB — 4 points (BL,BR,TR,TL), then 1 line click",
            10,
            30,
        )
        draw_label(
            vis,
            "RMB — reset; Backspace — undo; N — next; S — save; Q/Esc — quit",
            10,
            60,
        )

        for i, (x, y) in enumerate(st.src):
            cv2.circle(vis, (x, y), 6, (255, 255, 0), -1, lineType=cv2.LINE_AA)
            draw_label(vis, f"{i+1}", x + 8, y + 14)
        if len(st.src) == 4:
            poly = np.array(st.src, dtype=int)
            cv2.polylines(vis, [poly], True, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        if st.line_pt is not None and len(st.src) == 4:
            xL, _ = st.src[0]
            xR, _ = st.src[1]
            y = st.line_pt[1]
            cv2.line(vis, (xL, y), (xR, y), (0, 220, 255), 3, lineType=cv2.LINE_AA)

        cv2.imshow(WIN, vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(WIN)
            print("[INFO] Quit without saving.")
            return
        if key == 8:  # Backspace
            if st.line_pt is not None:
                st.line_pt = None
            elif st.src:
                st.src.pop()
        if key in (ord("n"), ord("N")):
            if len(st.src) != 4 or st.line_pt is None:
                print("[WARN] Need 4 points and a line for current ROI.")
            else:
                idx = min(idx + 1, n_rois - 1)
                print(f"[NEXT] ROI {idx+1}/{n_rois}")
        if key in (ord("s"), ord("S")):
            ok_all = all(len(s.src) == 4 and s.line_pt is not None for s in states)
            if not ok_all:
                print("[WARN] Not all ROIs are annotated.")
                continue
            cfg = load_config(cfg_path)
            cfg_rois = []
            for i, s in enumerate(states):
                base = (
                    DEFAULT_CFG["rois"][i]
                    if i < len(DEFAULT_CFG["rois"])
                    else DEFAULT_CFG["rois"][0]
                )
                width_m = float(base.get("width_m", 14.4))
                length_m = float(base.get("length_m", 100.0))
                H = homography_px_to_meters(
                    np.array(s.src, np.float32), width_m, length_m
                )
                y_m = float(persp_points(H, np.array([s.line_pt], np.float32))[0][1])
                cfg_rois.append(
                    {
                        "name": base.get("name", f"dir{i+1}"),
                        "src": s.src,
                        "width_m": width_m,
                        "length_m": length_m,
                        "count_line_y_m": y_m,
                        "flow_direction": base.get("flow_direction", "forward"),
                    }
                )
            cfg["rois"] = cfg_rois
            save_config(cfg_path, cfg)
            cv2.destroyWindow(WIN)
            print(f"[OK] Saved: {cfg_path}")
            for r in cfg_rois:
                print(f"  - {r['name']}: line_y_m={r['count_line_y_m']:.2f}")
            return


# ---------- Main pipeline ----------
def run(video_path: str, cfg: dict, device: str | None):
    os.makedirs(os.path.dirname(cfg["output_path"]) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {video_path}")
            sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    Wf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(
        cfg["output_path"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wf, Hf)
    )

    # Precompute homographies/lines for all ROIs
    rois = []
    palette = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
    for i, r in enumerate(cfg["rois"]):
        src = np.array(r["src"], np.float32)
        w_m, l_m = float(r["width_m"]), float(r["length_m"])
        H_px2m = homography_px_to_meters(src, w_m, l_m)
        H_m2px = np.linalg.inv(H_px2m)
        rois.append(
            {
                "name": r.get("name", f"dir{i+1}"),
                "poly": src.astype(int),
                "H_px2m": H_px2m,
                "H_m2px": H_m2px,
                "width_m": w_m,
                "length_m": l_m,
                "line_y_m": float(r["count_line_y_m"]),
                "flow": r.get("flow_direction", "forward").lower(),
                "color": palette[i % len(palette)],
            }
        )

    # Model (YOLO + ByteTrack)
    model = YOLO("yolo11n.pt")
    results_gen = model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True,
        classes=cfg["classes"],
        conf=float(cfg["conf"]),
        iou=float(cfg["iou"]),
        device=device,
        verbose=False,
    )

    # Histories/counters
    hist: Dict[Tuple[int, int], Deque[Tuple[int, float, float]]] = defaultdict(
        lambda: deque(maxlen=256)
    )
    smoothing_N = max(2, int(cfg["speed_smoothing_sec"] * fps))
    min_frames = int(cfg["min_track_frames"])
    limit_kmh = float(cfg["speed_limit_kmh"])

    counted_ids = [set() for _ in rois]
    overspeed_ids = [set() for _ in rois]
    overspeed_by_class = [defaultdict(int) for _ in rois]
    total_passed = [0 for _ in rois]

    def draw_overlay(frame):
        # draw ROI polygons + counting lines + small stats
        for i, r in enumerate(rois):
            cv2.polylines(frame, [r["poly"]], True, r["color"], 2, lineType=cv2.LINE_AA)
            line_m = np.array(
                [[0, r["line_y_m"]], [r["width_m"], r["line_y_m"]]], np.float32
            )
            line_px = persp_points(r["H_m2px"], line_m).astype(int)
            cv2.line(
                frame,
                tuple(line_px[0]),
                tuple(line_px[1]),
                (0, 220, 255),
                3,
                lineType=cv2.LINE_AA,
            )
            draw_label(
                frame,
                f"{r['name']} limit={limit_kmh:.0f} dir={r['flow']}",
                10,
                30 + 30 * i,
            )
        draw_label(
            frame,
            f"passed total: {sum(total_passed)} | overspeed total: {sum(len(s) for s in overspeed_ids)}",
            10,
            Hf - 10,
        )

    frame_idx = -1
    for res in results_gen:
        ok, frame = cap.read()
        frame_idx += 1
        if not ok:
            break

        draw_overlay(frame)

        if res is None or res.boxes is None or len(res.boxes) == 0:
            writer.write(frame)
            continue

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_id = boxes.cls.cpu().numpy().astype(int)
        tids = (
            boxes.id.cpu().numpy().astype(int)
            if getattr(boxes, "id", None) is not None
            else None
        )

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tid = int(tids[i]) if (tids is not None and i < len(tids)) else None
            cname = COCO_MAP.get(int(cls_id[i]), str(int(cls_id[i])))

            # pick ROI by bbox intersection (center OR corners)
            roi_idx = -1
            for j, r in enumerate(rois):
                if bbox_hits_poly(x1, y1, x2, y2, cx, cy, r["poly"], pad=4):
                    roi_idx = j
                    break
            if roi_idx < 0:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (160, 160, 160), 1, lineType=cv2.LINE_AA
                )
                continue

            r = rois[roi_idx]
            xm, ym = persp_points(r["H_px2m"], np.array([[cx, cy]], np.float32))[0]
            xm, ym = float(xm), float(ym)

            # update history & compute speed if enough points
            speed_kmh = 0.0
            if tid is not None:
                key = (roi_idx, tid)
                hist[key].append((frame_idx, xm, ym))
                if len(hist[key]) >= max(min_frames, 2):
                    last = list(hist[key])[-smoothing_N:]
                    speed_kmh = robust_speed_kmh(last, fps)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), r["color"], 2, lineType=cv2.LINE_AA
            )
            draw_label(
                frame,
                f"{r['name']} #{tid if tid is not None else '-'} {cname} {speed_kmh:.1f} km/h",
                x1,
                y1,
            )

            # single counting on line crossing (per ROI)
            if (
                tid is not None
                and tid not in counted_ids[roi_idx]
                and len(hist[(roi_idx, tid)]) >= 2
            ):
                prev_ym = hist[(roi_idx, tid)][-2][2]
                now_ym = hist[(roi_idx, tid)][-1][2]
                if crossed(prev_ym, now_ym, r["line_y_m"], r["flow"]):
                    counted_ids[roi_idx].add(tid)
                    total_passed[roi_idx] += 1
                    if speed_kmh > limit_kmh:
                        overspeed_ids[roi_idx].add(tid)
                        overspeed_by_class[roi_idx][cname] += 1
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            3,
                            lineType=cv2.LINE_AA,
                        )

        writer.write(frame)

    cap.release()
    writer.release()

    print("\n=== SUMMARY (multi-ROI) =======================")
    print(f"Video: {video_path}")
    print(f"Output: {cfg['output_path']}")
    print(f"FPS: {fps:.2f}, Frames: {total_frames}")
    for i, r in enumerate(rois):
        print(
            f"\n[{r['name']}] dir={r['flow']} width={r['width_m']}m length={r['length_m']}m line_y={r['line_y_m']:.1f}m"
        )
        print(f"  passed: {total_passed[i]} | overspeed: {len(overspeed_ids[i])}")
    print("==============================================\n")


# ---------- CLI ----------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="traffic.mp4")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--setup",
        type=int,
        nargs="?",
        const=1,
        help="interactive setup: specify number of ROIs (directions), e.g., --setup 2",
    )
    args = ap.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    if args.setup is not None:
        n = max(1, int(args.setup))
        setup_interactive(video_path, args.config, n_rois=n)
        return

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(cfg["output_path"]) or ".", exist_ok=True)
    run(video_path, cfg, args.device)


if __name__ == "__main__":
    main()
