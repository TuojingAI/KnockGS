"""Five-feature extraction from 31 aligned simulator-exported particle frames.

Ported from the experiment implementation without changing the descriptor.
Velocity is a per-frame displacement proxy, not displacement per second.
"""
from pathlib import Path
import numpy as np

EPS = 1e-12
FEATURES = ['norm_auc_rms_disp', 'norm_initial_slope_rms_0_5', 'norm_peak_rms_velocity_proxy', 'norm_auc_rms_velocity_proxy', 'early_velocity_ratio']

def frame_paths(case_root: Path):
    return sorted((case_root / "physical_state").glob("frame_*.npz"))

def load_positions(path: Path):
    with np.load(path) as data:
        key = "positions" if "positions" in data else list(data.keys())[0]
        return np.asarray(data[key], dtype=np.float32)

def extract_five_features(case_root: Path):
    frames = frame_paths(case_root)
    if len(frames) != 31:
        raise RuntimeError(f"Expected 31 frames under {case_root}, got {len(frames)}")
    pos0 = load_positions(frames[0])
    bbox_diag0 = float(np.linalg.norm(np.max(pos0, axis=0) - np.min(pos0, axis=0)))
    if bbox_diag0 <= EPS:
        raise RuntimeError(f"Degenerate initial bounding box: {case_root}")

    previous = pos0
    rms_disp_curve = []
    rms_vel_curve = []
    axis_rms_curves = [[], [], []]
    centroid_curve = []
    for index, path in enumerate(frames):
        pos = pos0 if index == 0 else load_positions(path)
        disp = pos - pos0
        rms_disp_curve.append(float(np.sqrt(np.mean(np.sum(disp * disp, axis=1)))))
        for axis in range(3):
            axis_rms_curves[axis].append(float(np.sqrt(np.mean(disp[:, axis] ** 2))))
        centroid_curve.append(np.mean(disp, axis=0).astype(np.float64))
        if index == 0:
            rms_vel_curve.append(0.0)
        else:
            delta = pos - previous
            rms_vel_curve.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
        previous = pos

    i5 = min(5, len(rms_disp_curve) - 1)
    initial_slope = (rms_disp_curve[i5] - rms_disp_curve[0]) / max(i5, 1)
    vel_total = float(sum(rms_vel_curve))
    features = {
        "num_frames": len(frames),
        "num_particles": int(pos0.shape[0]),
        "bbox_diag0": bbox_diag0,
        "norm_auc_rms_disp": float(sum(rms_disp_curve) / bbox_diag0),
        "norm_initial_slope_rms_0_5": float(initial_slope / bbox_diag0),
        "norm_peak_rms_velocity_proxy": float(max(rms_vel_curve) / bbox_diag0),
        "norm_auc_rms_velocity_proxy": float(vel_total / bbox_diag0),
        "early_velocity_ratio": float(
            sum(rms_vel_curve[1 : min(6, len(rms_vel_curve))]) / max(vel_total, EPS)
        ),
        "rms_disp_curve": ";".join(f"{x:.10g}" for x in rms_disp_curve),
        "rms_velocity_curve": ";".join(f"{x:.10g}" for x in rms_vel_curve),
        "axis_x_rms_curve": ";".join(f"{x:.10g}" for x in axis_rms_curves[0]),
        "axis_y_rms_curve": ";".join(f"{x:.10g}" for x in axis_rms_curves[1]),
        "axis_z_rms_curve": ";".join(f"{x:.10g}" for x in axis_rms_curves[2]),
        "centroid_x_curve": ";".join(f"{x[0]:.10g}" for x in centroid_curve),
        "centroid_y_curve": ";".join(f"{x[1]:.10g}" for x in centroid_curve),
        "centroid_z_curve": ";".join(f"{x[2]:.10g}" for x in centroid_curve),
    }
    return features
