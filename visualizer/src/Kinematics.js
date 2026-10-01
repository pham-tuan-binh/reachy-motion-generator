/**
 * Reachy Mini Kinematics - JavaScript port of kinematics-wasm
 *
 * Calculates passive joints for the Stewart platform from head joints and head pose.
 * Ported from: https://github.com/pollen-robotics/reachy-mini-desktop-app/tree/develop/kinematics-wasm
 */

// Constants from kinematics_data.json and URDF
const HEAD_Z_OFFSET = 0.177;
const MOTOR_ARM_LENGTH = 0.04;

// XL330 frame pose in head frame (from URDF)
const T_HEAD_XL_330 = [
    [0.4822, -0.7068, -0.5177, 0.0206],
    [0.1766, -0.5003, 0.8476, -0.0218],
    [-0.8581, -0.5001, -0.1164, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// Passive joint orientation offsets (from URDF)
const PASSIVE_ORIENTATION_OFFSET = [
    [-0.13754, -0.0882156, 2.10349],
    [-Math.PI, 5.37396e-16, -Math.PI],
    [0.373569, 0.0882156, -1.0381],
    [-0.0860846, 0.0882156, 1.0381],
    [0.123977, 0.0882156, -1.0381],
    [3.0613, 0.0882156, 1.0381],
    [Math.PI, 2.10388e-17, 4.15523e-17],
];

// Stewart rod direction in passive frame (from URDF)
const STEWART_ROD_DIR_IN_PASSIVE_FRAME = [
    [1.0, 0.0, 0.0],
    [0.50606941, -0.85796418, -0.08826792],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
];

// Motor data from kinematics_data.json
const MOTORS = [
    // stewart_1
    {
        branchPosition: [0.020648178337122566, 0.021763723638894568, 1.0345743467476964e-07],
        tWorldMotor: [
            [0.8660247915798899, 0.0000044901959360, -0.5000010603477224, 0.0269905781109381],
            [-0.5000010603626028, 0.0000031810770988, -0.8660247915770969, 0.0267489144601032],
            [-0.0000022980790772, 0.9999999999848599, 0.0000049999943606, 0.0766332540902687],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    // stewart_2
    {
        branchPosition: [0.00852381571767217, 0.028763668526131346, 1.183437210727778e-07],
        tWorldMotor: [
            [-0.8660211183436273, -0.0000044902196459, -0.5000074225075980, 0.0096699703080478],
            [0.5000074225224782, -0.0000031810634097, -0.8660211183408341, 0.0367490037948058],
            [0.0000022980697230, -0.9999999999848597, 0.0000050000112432, 0.0766333000521544],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    // stewart_3
    {
        branchPosition: [-0.029172011376922807, 0.0069999429399361995, 4.0290270064691214e-08],
        tWorldMotor: [
            [0.0000063267948970, -0.0000010196153098, 0.9999999999794665, -0.0366606982562266],
            [0.9999999999799865, 0.0000000000135060, -0.0000063267948965, 0.0100001160862987],
            [-0.0000000000070551, 0.9999999999994809, 0.0000010196153103, 0.0766334229944826],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    // stewart_4
    {
        branchPosition: [-0.029172040355214434, -0.0069999960097160766, -3.1608172912367394e-08],
        tWorldMotor: [
            [-0.0000036732050704, 0.0000010196153103, 0.9999999999927344, -0.0366607717202358],
            [-0.9999999999932538, -0.0000000000036776, -0.0000036732050700, -0.0099998653384376],
            [-0.0000000000000677, -0.9999999999994809, 0.0000010196153103, 0.0766334229944823],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    // stewart_5
    {
        branchPosition: [0.008523809101930114, -0.028763713010385224, -1.4344916837716326e-07],
        tWorldMotor: [
            [-0.8660284647694136, 0.0000044901728834, -0.4999946981608615, 0.0096697448698383],
            [-0.4999946981757425, -0.0000031811099295, 0.8660284647666202, -0.0367490491228644],
            [0.0000022980794298, 0.9999999999848597, 0.0000049999943840, 0.0766333000520353],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    // stewart_6
    {
        branchPosition: [0.020648186722822436, -0.02176369606185343, -8.957920105689965e-08],
        tWorldMotor: [
            [0.8660247915798903, -0.0000044901962204, -0.5000010603477218, 0.0269903370664035],
            [0.5000010603626028, 0.0000031810964559, 0.8660247915770964, -0.0267491384573748],
            [-0.0000022980696448, -0.9999999999848597, 0.0000050000112666, 0.0766332540903862],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
];

// ============= Matrix/Vector Math Utilities =============

/**
 * Create 3x3 rotation matrix from euler angles (xyz intrinsic = Z * Y * X matrix order)
 * Matches scipy's R.from_euler('xyz', angles)
 */
function rotationFromEulerXYZ(x, y, z) {
    const cx = Math.cos(x), sx = Math.sin(x);
    const cy = Math.cos(y), sy = Math.sin(y);
    const cz = Math.cos(z), sz = Math.sin(z);

    // Intrinsic xyz = Rz * Ry * Rx
    return [
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ];
}

/**
 * Extract euler angles (XYZ order) from 3x3 rotation matrix
 */
function eulerFromRotationXYZ(r) {
    const sy = r[0][2];

    if (Math.abs(sy) < 0.99999) {
        const x = Math.atan2(-r[1][2], r[2][2]);
        const y = Math.asin(sy);
        const z = Math.atan2(-r[0][1], r[0][0]);
        return [x, y, z];
    } else {
        // Gimbal lock
        const x = Math.atan2(r[2][1], r[1][1]);
        const y = sy > 0 ? Math.PI / 2 : -Math.PI / 2;
        const z = 0;
        return [x, y, z];
    }
}

/**
 * Multiply two 3x3 matrices
 */
function mat3Multiply(a, b) {
    const result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            for (let k = 0; k < 3; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

/**
 * Transpose a 3x3 matrix
 */
function mat3Transpose(m) {
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ];
}

/**
 * Multiply 3x3 matrix by 3D vector
 */
function mat3Vec3Multiply(m, v) {
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ];
}

/**
 * Add two 3D vectors
 */
function vec3Add(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

/**
 * Subtract two 3D vectors
 */
function vec3Sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/**
 * Normalize a 3D vector
 */
function vec3Normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-10) return [0, 0, 0];
    return [v[0] / len, v[1] / len, v[2] / len];
}

/**
 * Dot product of two 3D vectors
 */
function vec3Dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Cross product of two 3D vectors
 */
function vec3Cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

/**
 * Vector length
 */
function vec3Length(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/**
 * Scale a vector
 */
function vec3Scale(v, s) {
    return [v[0] * s, v[1] * s, v[2] * s];
}

/**
 * 3x3 identity matrix
 */
function mat3Identity() {
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
}

/**
 * Add two 3x3 matrices
 */
function mat3Add(a, b) {
    return [
        [a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
        [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]],
    ];
}

/**
 * Scale a 3x3 matrix
 */
function mat3Scale(m, s) {
    return [
        [m[0][0] * s, m[0][1] * s, m[0][2] * s],
        [m[1][0] * s, m[1][1] * s, m[1][2] * s],
        [m[2][0] * s, m[2][1] * s, m[2][2] * s],
    ];
}

/**
 * Create skew-symmetric matrix from vector (for cross product as matrix multiply)
 */
function skewSymmetric(v) {
    return [
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ];
}

/**
 * Align vectors: find rotation that aligns 'from' to 'to'
 * Similar to scipy.spatial.transform.Rotation.align_vectors
 */
function alignVectors(from, to) {
    const fromN = vec3Normalize(from);
    const toN = vec3Normalize(to);

    const dot = vec3Dot(fromN, toN);

    // Vectors are nearly parallel
    if (dot > 0.99999) {
        return mat3Identity();
    }

    // Vectors are nearly opposite
    if (dot < -0.99999) {
        // Find a perpendicular axis
        let perp = vec3Cross([1, 0, 0], fromN);
        if (vec3Length(perp) < 0.001) {
            perp = vec3Cross([0, 1, 0], fromN);
        }
        const axis = vec3Normalize(perp);
        // Rotate 180 degrees around perpendicular axis
        const k = skewSymmetric(axis);
        const k2 = mat3Multiply(k, k);
        return mat3Add(mat3Identity(), mat3Scale(k2, 2));
    }

    // General case: Rodrigues' rotation formula
    const cross = vec3Cross(fromN, toN);
    const s = vec3Length(cross);
    const c = dot;

    const k = skewSymmetric(cross);
    const k2 = mat3Multiply(k, k);

    // R = I + K + K^2 * (1 - c) / s^2
    const factor = (1 - c) / (s * s);
    return mat3Add(mat3Add(mat3Identity(), k), mat3Scale(k2, factor));
}

/**
 * Extract 3x3 rotation from 4x4 matrix
 */
function mat4ToRot3(m) {
    return [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
}

/**
 * Extract translation from 4x4 matrix
 */
function mat4ToTrans(m) {
    return [m[0][3], m[1][3], m[2][3]];
}

// ============= Main Kinematics Function =============

/**
 * Calculate passive joint angles from head joints and head pose
 *
 * @param {number[]} headJoints - Array of 7 floats: [yaw_body, stewart_1, ..., stewart_6]
 * @param {number[]} headPose - 4x4 transformation matrix as 16 floats (row-major)
 * @returns {number[]} Array of 21 floats: passive joint angles [p1_x, p1_y, p1_z, ..., p7_x, p7_y, p7_z]
 */
export function calculatePassiveJoints(headJoints, headPose) {
    if (!headJoints || headJoints.length < 7 || !headPose || headPose.length < 16) {
        return new Array(21).fill(0);
    }

    const bodyYaw = headJoints[0];

    // Build head pose matrix from row-major input
    const pose = [
        [headPose[0], headPose[1], headPose[2], headPose[3]],
        [headPose[4], headPose[5], headPose[6], headPose[7]],
        [headPose[8], headPose[9], headPose[10], headPose[11]],
        [headPose[12], headPose[13], headPose[14], headPose[15]],
    ];

    // Add head Z offset
    pose[2][3] += HEAD_Z_OFFSET;

    // Inverse rotation: rotate pose around Z by -body_yaw
    const cosYaw = Math.cos(bodyYaw);
    const sinYaw = Math.sin(bodyYaw);
    const rZInv = [
        [cosYaw, sinYaw, 0],
        [-sinYaw, cosYaw, 0],
        [0, 0, 1],
    ];

    // Apply inverse yaw rotation to pose
    const poseRot = mat4ToRot3(pose);
    const poseTrans = mat4ToTrans(pose);
    const rotatedPoseRot = mat3Multiply(rZInv, poseRot);
    const rotatedPoseTrans = mat3Vec3Multiply(rZInv, poseTrans);

    // Pre-compute passive correction rotations
    const passiveCorrections = PASSIVE_ORIENTATION_OFFSET.map(
        offset => rotationFromEulerXYZ(offset[0], offset[1], offset[2])
    );

    const passiveJoints = new Array(21).fill(0);
    let lastRServoBranch = mat3Identity();
    let lastRWorldServo = mat3Identity();

    // T_motor_servo_arm: translation by motor_arm_length along X
    const tMotorServoArm = [MOTOR_ARM_LENGTH, 0, 0];

    // For each of the 6 stewart motors
    for (let i = 0; i < 6; i++) {
        const motor = MOTORS[i];
        const stewartJoint = headJoints[i + 1];

        // Calculate branch position on platform in world frame
        const branchPosWorld = vec3Add(
            mat3Vec3Multiply(rotatedPoseRot, motor.branchPosition),
            rotatedPoseTrans
        );

        // Compute servo rotation (rotating around Z axis)
        const cosZ = Math.cos(stewartJoint);
        const sinZ = Math.sin(stewartJoint);
        const rServo = [
            [cosZ, -sinZ, 0],
            [sinZ, cosZ, 0],
            [0, 0, 1],
        ];

        // T_world_motor rotation and translation
        const tWorldMotorRot = mat4ToRot3(motor.tWorldMotor);
        const tWorldMotorTrans = mat4ToTrans(motor.tWorldMotor);

        // Compute world servo arm position
        const servoPosLocal = mat3Vec3Multiply(rServo, tMotorServoArm);
        const pWorldServoArm = vec3Add(
            mat3Vec3Multiply(tWorldMotorRot, servoPosLocal),
            tWorldMotorTrans
        );

        // Apply passive correction to orientation
        const rWorldServo = mat3Multiply(
            mat3Multiply(tWorldMotorRot, rServo),
            passiveCorrections[i]
        );

        // Vector from servo arm to branch in world frame
        const vecServoToBranch = vec3Sub(branchPosWorld, pWorldServoArm);

        // Transform to servo frame (use transpose for inverse of rotation)
        const vecServoToBranchInServo = mat3Vec3Multiply(
            mat3Transpose(rWorldServo),
            vecServoToBranch
        );

        // Rod direction in passive frame
        const rodDir = STEWART_ROD_DIR_IN_PASSIVE_FRAME[i];

        // Normalize and get straight line direction
        const straightLineDir = vec3Normalize(vecServoToBranchInServo);

        // Align rod direction to actual direction
        const rServoBranch = alignVectors(rodDir, straightLineDir);
        const euler = eulerFromRotationXYZ(rServoBranch);

        passiveJoints[i * 3] = euler[0];
        passiveJoints[i * 3 + 1] = euler[1];
        passiveJoints[i * 3 + 2] = euler[2];

        // Save for 7th passive joint calculation
        if (i === 5) {
            lastRServoBranch = rServoBranch;
            lastRWorldServo = rWorldServo;
        }
    }

    // 7th passive joint (XL330 on the head)
    const tHeadXl330Rot = mat4ToRot3(T_HEAD_XL_330);
    const rHeadXl330 = mat3Multiply(rotatedPoseRot, tHeadXl330Rot);

    // Current rod orientation with correction for 7th passive joint
    const rRodCurrent = mat3Multiply(
        mat3Multiply(lastRWorldServo, lastRServoBranch),
        passiveCorrections[6]
    );

    // Compute relative rotation
    const rDof = mat3Multiply(mat3Transpose(rRodCurrent), rHeadXl330);
    const euler7 = eulerFromRotationXYZ(rDof);

    passiveJoints[18] = euler7[0];
    passiveJoints[19] = euler7[1];
    passiveJoints[20] = euler7[2];

    return passiveJoints;
}

/**
 * Build head pose matrix from position and orientation
 * @param {Object} headPose - {x, y, z, roll, pitch, yaw}
 * @returns {number[]} 16-element row-major 4x4 matrix
 */
export function buildHeadPoseMatrix(headPose) {
    const { x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0 } = headPose;

    // Build rotation from euler angles (XYZ order)
    const rot = rotationFromEulerXYZ(roll, pitch, yaw);

    // Return as row-major 4x4 matrix
    return [
        rot[0][0], rot[0][1], rot[0][2], x,
        rot[1][0], rot[1][1], rot[1][2], y,
        rot[2][0], rot[2][1], rot[2][2], z,
        0, 0, 0, 1,
    ];
}

