"""Reachy Mini move format <-> 9-DoF trajectory arrays.

A *move* is the JSON format used by Pollen's emotion / dance libraries and the SDK's recorded-move
player: ``{"description", "time": [s], "set_target_data": [{"head": 4x4, "antennas": [r, l],
"body_yaw": rad}, ...]}``.

A *trajectory* is a (T, 9) float array sampled at ``FPS``:
``x y z (m) | roll pitch yaw (rad, extrinsic xyz) | antenna_right antenna_left (rad) | body_yaw (rad)``.
"""
import json
import numpy as np

FPS = 25
DOF = ["x", "y", "z", "roll", "pitch", "yaw", "antenna_right", "antenna_left", "body_yaw"]


def rpy(R):
    """Rotation matrices (..., 3, 3) -> extrinsic xyz roll/pitch/yaw (..., 3)."""
    sy = np.hypot(R[..., 0, 0], R[..., 1, 0])
    return np.stack([np.arctan2(R[..., 2, 1], R[..., 2, 2]), np.arctan2(-R[..., 2, 0], sy),
                     np.arctan2(R[..., 1, 0], R[..., 0, 0])], -1)


def rpy_to_mat(r):
    """(T, 3) roll/pitch/yaw -> (T, 3, 3) rotation matrices (inverse of ``rpy``)."""
    cr, cp, cy = np.cos(r).T
    sr, sp, sy = np.sin(r).T
    R = np.zeros((len(r), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp;     R[:, 2, 1] = cp * sr;                R[:, 2, 2] = cp * cr
    return R


def load_move(path):
    with open(path) as f:
        return json.load(f)


def traj(move, fps=FPS):
    """Move dict -> (T, 9) trajectory, resampled to ``fps`` using the move's own timestamps.
    (Library clips are recorded at ~100 Hz; playing their frames at 25 fps would be 4x too slow.)"""
    fr = move["set_target_data"]
    t = np.array(move["time"], float); t -= t[0]
    H = np.array([f["head"] for f in fr])
    A = np.concatenate([H[:, :3, 3], rpy(H[:, :3, :3]), np.array([f["antennas"] for f in fr]),
                        np.array([[f.get("body_yaw", 0.0)] for f in fr])], -1)
    dur = float(t[-1]) if t[-1] > 0 else len(fr) / fps
    u = np.linspace(0, dur, max(2, int(round(dur * fps))))
    return np.stack([np.interp(u, t, A[:, j]) for j in range(9)], -1)


def to_move(A, description="", fps=FPS):
    """(T, 9) trajectory -> move dict."""
    T = len(A)
    H = np.tile(np.eye(4), (T, 1, 1)); H[:, :3, :3] = rpy_to_mat(A[:, 3:6]); H[:, :3, 3] = A[:, :3]
    return dict(description=description, time=(np.arange(T) / fps).tolist(),
                set_target_data=[dict(head=H[i].tolist(), antennas=A[i, 6:8].tolist(), body_yaw=float(A[i, 8]),
                                      check_collision=False) for i in range(T)])


def mirror(A):
    """Sagittal mirror: negate y, roll, yaw, body yaw; swap AND negate the antennas
    (the right antenna droops with negative angles, the left with positive)."""
    B = A.copy(); B[:, [1, 3, 5, 8]] *= -1
    B[:, 6], B[:, 7] = -A[:, 7], -A[:, 6]
    return B


def stretch(A, f):
    """Uniform time-stretch by factor ``f`` (f > 1 = slower / longer)."""
    T = len(A); n = max(8, int(round(T * f))); u = np.linspace(0, T - 1, n)
    return np.stack([np.interp(u, np.arange(T), A[:, j]) for j in range(A.shape[1])], -1)
