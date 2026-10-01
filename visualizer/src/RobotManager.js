import * as THREE from 'three';
import URDFLoader from 'https://cdn.jsdelivr.net/npm/urdf-loader@0.12.3/+esm';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { calculatePassiveJoints, buildHeadPoseMatrix } from './Kinematics.js';

// Joint name constants
const HEAD_JOINT_NAMES = ['yaw_body', 'stewart_1', 'stewart_2', 'stewart_3', 'stewart_4', 'stewart_5', 'stewart_6'];
const PASSIVE_JOINT_NAMES = [];
for (let i = 1; i <= 7; i++) {
    PASSIVE_JOINT_NAMES.push(`passive_${i}_x`, `passive_${i}_y`, `passive_${i}_z`);
}

export class RobotManager {
    constructor(statusCallback, envMap = null) {
        this.robot = null;
        this.jointMap = {};
        this.statusCallback = statusCallback;
        this.envMap = envMap;

        // HuggingFace CDN base URL
        // Assets ship with this Space (originally from 8bitkick/reachy_mini_3d_web_viz, Apache-2.0)
        const HF_BASE = "./";

        // URL configuration
        const params = new URLSearchParams(location.search);
        this.URDF_URL = params.get("urdf") || HF_BASE + "assets/reachy-mini.urdf";
        this.STL_BASE_URL = params.get("stl_base") || "assets/";
        this.DEFAULT_STATE_URL = params.get("default_state") || HF_BASE + "assets/default_state.json";

        // HuggingFace CDN for mesh files (resolves LFS/Xet properly)
        this.HF_CDN_BASE = HF_BASE + "assets/meshes_optimized/";

        // Frame throttling (~30 Hz)
        this.lastUpdateTime = 0;
        this.updateInterval = 0;          // playback drives every frame itself
    }

    setEnvMap(envMap) {
        this.envMap = envMap;
    }

    async loadRobot() {
        this.statusCallback('Loading...');

        const response = await fetch(this.URDF_URL);
        const urdfText = await response.text();

        // Parse URDF to extract mesh-to-color mapping
        this.meshColors = this.parseUrdfColors(urdfText);

        const blob = new Blob([urdfText], { type: 'application/xml' });
        const blobUrl = URL.createObjectURL(blob);

        const loader = new URDFLoader();
        loader.packages = {
            'assets': this.STL_BASE_URL,
            'reachy_mini_description': this.STL_BASE_URL
        };
        loader.workingPath = this.STL_BASE_URL;

        // Custom mesh loader to support both STL and GLB formats
        const gltfLoader = new GLTFLoader();
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
        gltfLoader.setDRACOLoader(dracoLoader);
        const stlLoader = new STLLoader();

        loader.loadMeshCb = (path, manager, onComplete) => {
            const ext = path.split('.').pop().toLowerCase();
            // Get color for this mesh from our parsed URDF
            const filename = path.split('/').pop();
            const matData = this.meshColors[filename];

            // Use HuggingFace CDN for GLB files to properly resolve LFS/Xet
            if (ext === 'glb' || ext === 'gltf') {
                path = this.HF_CDN_BASE + filename;
            }


  // Create material with the URDF color and opacity
const opacity = matData?.opacity ?? 1;
const isTransparent = opacity < 0.4;

const material = new THREE.MeshPhysicalMaterial({
    color: (filename?.includes('antenna_V2') || isTransparent || matData?.name === 'antenna_material') ? 0x202020 : matData?.color,

    // Default: matte-ish non-metal
    metalness: 0.0,
    roughness: (filename?.includes('antenna_V2') || isTransparent || matData?.name === 'antenna_material') ? 0.05 : 0.7,

    transparent: isTransparent,
    opacity,
    side: isTransparent ? THREE.DoubleSide : THREE.FrontSide,
});

// Silver metal for 'link' meshes
if (filename?.includes('link')) {
  material.color.setHex(0xffffff);   // silver base
  material.metalness = 1.0;         // must be 1 for real metal
  material.roughness = 0.3;        // 0.1 = chrome-ish, 0.2 = brushed-ish
  material.needsUpdate = true;
}

// Shiny black for antenna
if (matData.name === 'antenna_material') {
  material.clearcoat = 1.0;
  material.clearcoatRoughness = 0.0;
  material.reflectivity = 1.0;
  material.envMapIntensity = 1.5;
  material.needsUpdate = true;
}

            if (ext === 'glb' || ext === 'gltf') {
                gltfLoader.load(path, (gltf) => {
                    let geometry = null;
                    gltf.scene.traverse((child) => {
                        if (child.isMesh && !geometry) {
                            geometry = child.geometry;
                        }
                    });
                    if (geometry) {
                        const mesh = new THREE.Mesh(geometry, material);
                        mesh.castShadow = true;
                        mesh.receiveShadow = true;
                        onComplete(mesh);
                    } else {
                        onComplete(gltf.scene);
                    }
                }, undefined, (err) => {
                    console.error('GLTFLoader error:', err);
                    onComplete(null, err);
                });
            } else {
                stlLoader.load(path, (geometry) => {
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    onComplete(mesh);
                }, undefined, (err) => {
                    console.error('STLLoader error:', err);
                    onComplete(null, err);
                });
            }
        };

        return new Promise((resolve, reject) => {
            loader.load(blobUrl, async (robot) => {
                this.robot = robot;
                this.robot.rotation.x = -Math.PI / 2;
                this.buildJointMap();
                this.applyMaterials();
                await this.loadDefaultState();
                URL.revokeObjectURL(blobUrl);
                resolve(this.robot);
            }, undefined, (err) => {
                URL.revokeObjectURL(blobUrl);
                reject(err);
            });
        });
    }

    buildJointMap() {
        this.robot.traverse((child) => {
            if (child.isURDFJoint) {
                this.jointMap[child.name] = child;
            }
        });
    }

    async loadDefaultState() {
        try {
            const response = await fetch(this.DEFAULT_STATE_URL);
            if (!response.ok) return;
            const defaultState = await response.json();
            // Force immediate update by resetting throttle
            this.lastUpdateTime = 0;
            this.updateJoints(defaultState);
        } catch (e) {
            console.warn('Could not load default state:', e);
        }
    }

    parseUrdfColors(urdfText) {
        // Parse URDF XML to extract mesh filename -> color mapping
        const parser = new DOMParser();
        const doc = parser.parseFromString(urdfText, 'application/xml');
        const colorMap = {};

        // Find all visual elements
        const visuals = doc.querySelectorAll('visual');
        visuals.forEach(visual => {
            const mesh = visual.querySelector('geometry mesh');
            const material = visual.querySelector('material color');

            if (mesh && material) {
                const filename = mesh.getAttribute('filename');
                const rgba = material.getAttribute('rgba');

                if (filename && rgba) {
                    // Extract just the filename without path
                    const name = filename.split('/').pop();
                    const [r, g, b, a] = rgba.split(' ').map(Number);
                    colorMap[name] = { color: new THREE.Color(r, g, b), opacity: a };
                }
            }
        });

        return colorMap;
    }

    applyMaterials() {
        // Materials are now applied during mesh loading with correct colors
        // This just ensures shadows are enabled on all meshes
        this.robot.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
            }
        });
    }

    updateJoints(data) {
        if (!this.robot) return;

        // Throttle updates
        const now = performance.now();
        if (now - this.lastUpdateTime < this.updateInterval) return;
        this.lastUpdateTime = now;

        // Get head pose matrix
        let headPoseMatrix = null;
        if (data.head_pose) {
            if (Array.isArray(data.head_pose) && data.head_pose.length === 16) {
                headPoseMatrix = data.head_pose;
            } else if (data.head_pose.m) {
                headPoseMatrix = data.head_pose.m;
            } else {
                headPoseMatrix = buildHeadPoseMatrix(data.head_pose);
            }
        }

        // Get head joints from WebSocket or use body_yaw fallback
        const headJoints = (data.head_joints?.length === 7)
            ? data.head_joints
            : [data.body_yaw || 0, 0, 0, 0, 0, 0, 0];

        // Apply head joints
        for (let i = 0; i < 7; i++) {
            const joint = this.jointMap[HEAD_JOINT_NAMES[i]];
            if (joint) joint.setJointValue(headJoints[i]);
        }

        // Calculate and apply passive joints
        if (headPoseMatrix) {
            const passiveJoints = calculatePassiveJoints(headJoints, headPoseMatrix);
            for (let i = 0; i < 21; i++) {
                const joint = this.jointMap[PASSIVE_JOINT_NAMES[i]];
                if (joint) joint.setJointValue(passiveJoints[i]);
            }
        }

        // Update antennas (swapped and inverted)
        if (data.antennas_position?.length >= 2) {
            this.jointMap['right_antenna']?.setJointValue(-data.antennas_position[0]);
            this.jointMap['left_antenna']?.setJointValue(-data.antennas_position[1]);
        }
    }
}
