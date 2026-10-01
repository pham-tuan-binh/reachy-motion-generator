/**
 * Stewart-platform inverse kinematics for Reachy Mini, in the browser.
 * A line-for-line port of pollen-robotics/reachy_mini_rust_kinematics (inverse_kinematics + inverse_kinematics_safe),
 * with the SDK's kinematics_data.json (Apache-2.0). Verified against the Rust implementation (see test_ik.mjs).
 *
 *   headJoints(pose4x4, bodyYaw) -> [bodyYaw', m1..m6]   (NaN motor angles = unreachable pose)
 */
const HEAD_Z_OFFSET = 0.177;
const RS = 0.04000000000000001;   // motor arm length
const RP = 0.08499999999999995;   // rod length
const MAX_RELATIVE_YAW = 65 * Math.PI / 180, MAX_BODY_YAW = 160 * Math.PI / 180;
const MOTORS = [{"T_motor_world": [[0.8660247915798898, -0.5000010603626028, -2.298079077119539e-06, -0.009999848080267933], [4.490195936008854e-06, 3.1810770986818273e-06, 0.999999999984859, -0.07663346037245178], [-0.500001060347722, -0.8660247915770963, 4.999994360718464e-06, 0.03666015757925319], [0.0, 0.0, 0.0, 1.0]], "branch": [0.020648178337122566, 0.021763723638894568, 1.0345743467476964e-07], "solution": -1}, {"T_motor_world": [[-0.8660211183436269, 0.5000074225224785, 2.298069723064582e-06, -0.01000055227585102], [-4.490219645842903e-06, -3.181063409649239e-06, -0.999999999984859, 0.07663346037219607], [-0.5000074225075973, -0.8660211183408337, 5.00001124330122e-06, 0.03666008712637943], [0.0, 0.0, 0.0, 1.0]], "branch": [0.00852381571767217, 0.028763668526131346, 1.183437210727778e-07], "solution": 1}, {"T_motor_world": [[6.326794896519466e-06, 0.9999999999799852, -7.0550646912150425e-12, -0.009999884140839245], [-1.0196153102346142e-06, 1.3505961633338446e-11, 0.9999999999994795, -0.07663346037438698], [0.9999999999794655, -6.326794896940104e-06, 1.0196153098685706e-06, 0.036660683387545835], [0.0, 0.0, 0.0, 1.0]], "branch": [-0.029172011376922807, 0.0069999429399361995, 4.0290270064691214e-08], "solution": -1}, {"T_motor_world": [[-3.673205069955933e-06, -0.9999999999932537, -6.767968877969483e-14, -0.010000000000897517], [1.0196153102837198e-06, -3.6775764393585005e-12, -0.9999999999994795, 0.0766334603742898], [0.9999999999927336, -3.673205070385213e-06, 1.0196153102903487e-06, 0.03666065685180194], [0.0, 0.0, 0.0, 1.0]], "branch": [-0.029172040355214434, -0.0069999960097160766, -3.1608172912367394e-08], "solution": 1}, {"T_motor_world": [[-0.8660284647694133, -0.4999946981757419, 2.298079429767357e-06, -0.010000231529504576], [4.490172883391843e-06, -3.1811099293773187e-06, 0.9999999999848591, -0.07663346037246624], [-0.4999946981608617, 0.8660284647666201, 4.999994384073154e-06, 0.03666016059492482], [0.0, 0.0, 0.0, 1.0]], "branch": [0.008523809101930114, -0.028763713010385224, -1.4344916837716326e-07], "solution": -1}, {"T_motor_world": [[0.8660247915798897, 0.5000010603626025, -2.298069644866714e-06, -0.009999527331574583], [-4.490196220318687e-06, 3.1810964558725514e-06, -0.9999999999848591, 0.07663346037272492], [-0.500001060347722, 0.8660247915770967, 5.000011266610794e-06, 0.036660231042625266], [0.0, 0.0, 0.0, 1.0]], "branch": [0.020648186722822436, -0.02176369606185343, -8.957920105689965e-08], "solution": 1}];

const matmul = (A, B) => A.map((r, i) => B[0].map((_, j) => r.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const rotZ = (a) => [[Math.cos(a), -Math.sin(a), 0, 0], [Math.sin(a), Math.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

function motorAngles(T, bodyYaw) {
  const Tt = matmul(rotZ(-bodyYaw), T);
  return MOTORS.map(({T_motor_world, branch, solution}) => {
    const Tb = [[1, 0, 0, branch[0]], [0, 1, 0, branch[1]], [0, 0, 1, branch[2]], [0, 0, 0, 1]];
    const M = matmul(matmul(T_motor_world, Tt), Tb);
    const px = M[0][3], py = M[1][3], pz = M[2][3];
    const px2 = px * px, py2 = py * py, pz2 = pz * pz, rs2 = RS * RS, rp2 = RP * RP;
    const x = px2 + 2 * px * RS + py2 + pz2 - rp2 + rs2;
    const disc = -(px2 * px2) - 2 * px2 * py2 - 2 * px2 * pz2 + 2 * px2 * rp2 + 2 * px2 * rs2 - py2 * py2 - 2 * py2 * pz2
      + 2 * py2 * rp2 + 2 * py2 * rs2 - pz2 * pz2 + 2 * pz2 * rp2 - 2 * pz2 * rs2 - rp2 * rp2 + 2 * rp2 * rs2 - rs2 * rs2;
    const y = 2 * py * RS + solution * Math.sqrt(disc);
    return wrap(2 * Math.atan2(y, x));
  });
}

/** pose: 4x4 head pose (SDK convention, as in recorded moves); bodyYaw in radians. */
export function headJoints(pose, bodyYaw = 0) {
  const T = pose.map((r) => r.slice()); T[2][3] += HEAD_Z_OFFSET;
  let target = -bodyYaw;
  const currentYaw = Math.atan2(T[0][1], T[0][0]);
  target = currentYaw + Math.max(-MAX_RELATIVE_YAW, Math.min(MAX_RELATIVE_YAW, target - currentYaw));
  target = -Math.max(-MAX_BODY_YAW, Math.min(MAX_BODY_YAW, target));
  return [target, ...motorAngles(T, target)];
}
