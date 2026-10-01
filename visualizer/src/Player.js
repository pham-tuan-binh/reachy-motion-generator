import { headJoints } from './StewartIK.js';

/**
 * Plays Reachy Mini moves (SDK recorded-move format) on the 3D robot.
 * Each frame: head_pose (row-major 4x4) + antennas + body_yaw -> motor angles via the browser IK -> RobotManager.
 */
export class Player {
    constructor(robotManager, onTick) {
        this.robot = robotManager; this.onTick = onTick;
        this.frames = null; this.times = null; this.t0 = 0; this.pausedAt = null;
        this.speed = 1; this.loop = true;
        requestAnimationFrame(this.tick.bind(this));
    }

    /** Precompute the viewer state for every frame of a move (IK runs once, not per render). */
    load(move) {
        const fr = move.set_target_data, t = move.time;
        const t0 = t[0];
        this.times = t.map((x) => x - t0);
        this.frames = fr.map((f) => ({
            head_pose: f.head.flat(),
            head_joints: headJoints(f.head, f.body_yaw ?? 0),
            antennas_position: f.antennas,
        }));
        // an unreachable frame (NaN motor angle) holds the previous valid one, like the daemon does
        for (let i = 1; i < this.frames.length; i++) {
            if (this.frames[i].head_joints.some(Number.isNaN)) this.frames[i] = this.frames[i - 1];
        }
        this.t0 = performance.now(); this.pausedAt = null;
        return this.duration();
    }

    duration() { return this.times ? this.times[this.times.length - 1] : 0; }
    playing() { return this.frames && this.pausedAt === null; }
    toggle() {
        if (!this.frames) return;
        if (this.pausedAt === null) this.pausedAt = this.elapsed();
        else { this.t0 = performance.now() - this.pausedAt * 1000 / this.speed; this.pausedAt = null; }
    }
    restart() { this.t0 = performance.now(); this.pausedAt = null; }
    setSpeed(s) { const e = this.elapsed(); this.speed = s; this.t0 = performance.now() - e * 1000 / s; }
    seek(frac) {
        const e = frac * this.duration();
        if (this.pausedAt !== null) this.pausedAt = e; else this.t0 = performance.now() - e * 1000 / this.speed;
    }
    elapsed() {
        if (!this.frames) return 0;
        if (this.pausedAt !== null) return this.pausedAt;
        let e = (performance.now() - this.t0) / 1000 * this.speed;
        const d = this.duration();
        if (e > d) { if (this.loop) { e = e % (d + 0.6); if (e > d) e = d; } else e = d; }   // brief rest between loops
        return e;
    }

    tick() {
        if (this.frames) {
            const e = this.elapsed();
            let i = this.times.findIndex((x) => x > e); if (i < 0) i = this.times.length; i = Math.max(0, i - 1);
            this.robot.updateJoints(this.frames[i]);
            this.onTick?.(e, this.duration());
        }
        requestAnimationFrame(this.tick.bind(this));
    }
}
