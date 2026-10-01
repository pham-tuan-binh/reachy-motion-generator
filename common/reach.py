"""Project a move onto the Stewart platform's reachable set, frame by frame.

The reachable set is coupled in task space (a tilt reachable at one height is not at another), so no
per-channel clipping can guarantee it. The daemon rejects unreachable targets and holds the last one,
which looks like a freeze on hardware. Instead, for each unreachable head pose we line-search from the
last reachable pose toward it (SE(3) interpolation, body yaw interpolated with it) and keep the
furthest reachable point. Antennas and timing are untouched.

Uses the SDK's analytical IK (``reachy_mini_rust_kinematics``) with the same limits as the daemon.
"""
import json
from pathlib import Path

import numpy as np
from reachy_mini_rust_kinematics import ReachyMiniRustKinematics

KINEMATICS = Path(__file__).parent / "assets" / "kinematics_data.json"   # from pollen-robotics/reachy_mini (Apache-2.0)


class Reach:
    def __init__(self, kinematics_path=KINEMATICS):
        d = json.load(open(kinematics_path, "rb")); self.z = d["head_z_offset"]
        self.kin = ReachyMiniRustKinematics(d["motor_arm_length"], d["rod_length"])
        for m in d["motors"]:
            self.kin.add_branch(m["branch_position"], np.linalg.inv(m["T_motor_world"]), 1 if m["solution"] else -1)

    def ik(self, H, yaw):
        """Head pose (4x4) + body yaw -> 7 joint targets [body_yaw, stewart_1..6] (NaN if unreachable)."""
        p = np.array(H, float); p[2, 3] += self.z
        return np.array(self.kin.inverse_kinematics_safe(p.tolist(), body_yaw=float(yaw),
                                                         max_relative_yaw=np.deg2rad(65), max_body_yaw=np.deg2rad(160)))

    def ok(self, H, yaw):
        return bool(np.all(np.isfinite(self.ik(H, yaw))))

    @staticmethod
    def interp(H0, H1, a):
        """SE(3) interpolation: linear position, geodesic (slerp) rotation."""
        R0, R1 = H0[:3, :3], H1[:3, :3]; Rrel = R0.T @ R1
        ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
        if ang < 1e-8:
            Ra = R0
        else:
            w = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / (2 * np.sin(ang))
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]); t = a * ang
            Ra = R0 @ (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K)
        H = np.eye(4); H[:3, :3] = Ra; H[:3, 3] = (1 - a) * H0[:3, 3] + a * H1[:3, 3]
        return H

    def project(self, move, iters=12):
        """In place. Returns (move, fraction of frames that had to be projected)."""
        fr = move["set_target_data"]; last = None; fixed = 0
        for f in fr:
            H = np.array(f["head"], float); yaw = float(f.get("body_yaw", 0.0))
            if self.ok(H, yaw):
                last = (H, yaw); continue
            fixed += 1
            H0, y0 = last if last is not None else (np.eye(4), 0.0)
            lo, hi = 0.0, 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if self.ok(self.interp(H0, H, mid), (1 - mid) * y0 + mid * yaw): lo = mid
                else: hi = mid
            Hp, yp = self.interp(H0, H, lo), (1 - lo) * y0 + lo * yaw
            f["head"] = Hp.tolist(); f["body_yaw"] = yp; last = (Hp, yp)
        return move, fixed / max(1, len(fr))
