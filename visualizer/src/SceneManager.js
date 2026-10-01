import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class SceneManager {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.envMap = null;

        this.init();
    }

    init() {
        // Scene setup
        this.scene = new THREE.Scene();

        // Camera setup - positioned for good robot view
        this.camera = new THREE.PerspectiveCamera(
            50,
            window.innerWidth / window.innerHeight,
            0.01,
            100
        );
        this.camera.position.set(0.4, 0.1, -0.4);
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 2;
        this.controls.target.set(0, 0.15, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 0.2;
        this.controls.maxDistance = 2;
        this.controls.update();

        // Three-point lighting setup (based on reference implementation)
        this.setupLighting();

        // Ground grid
        this.addGroundGrid();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        // Ambient light - soft base illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        // Key light - main directional light with shadows
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
        keyLight.position.set(2, 1, 2);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 1024;
        keyLight.shadow.mapSize.height = 1024;
        keyLight.shadow.camera.near = 0.1;
        keyLight.shadow.camera.far = 10;
        keyLight.shadow.camera.left = -1;
        keyLight.shadow.camera.right = 1;
        keyLight.shadow.camera.top = 1;
        keyLight.shadow.camera.bottom = -1;
        this.scene.add(keyLight);

        // Fill light - softer, opposite side
        const fillLight = new THREE.DirectionalLight(0xFFB366, 0.6);
        fillLight.position.set(-2, 0.5, 1.5);
        this.scene.add(fillLight);

        // Rim/back light - warm tone for depth
        const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
        rimLight.position.set(0, 1.2, -2);
        this.scene.add(rimLight);

        // Hemisphere light for natural ambient
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
        hemiLight.position.set(0, 1, 0);
        this.scene.add(hemiLight);

        // Create environment map from gradient colors (matches CSS background)
        // this.createEnvMap();

    }

    createEnvMap() {
        // Load an environment map from a reference JPG on the internet
        const loader = new THREE.TextureLoader();
        loader.load(
            'https://threejs.org/examples/textures/2294472375_24a3b8ef46_o.jpg', // Example equirectangular image
            (texture) => {
            texture.mapping = THREE.EquirectangularReflectionMapping;
            texture.colorSpace = THREE.SRGBColorSpace;
            this.envMap = texture;
            this.scene.environment = this.envMap;
            },
            undefined,
            (err) => {
            console.error('Failed to load environment map:', err);
            }
        );
    }   


    addGroundGrid() {
        // Ground plane for shadows
        const groundGeometry = new THREE.PlaneGeometry(2, 2);
        const groundMaterial = new THREE.ShadowMaterial({ opacity: 0.3 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Grid helper
        const gridHelper = new THREE.GridHelper(1, 20, 0xbbbbbb, 0xbbbbbb);
        gridHelper.position.y = 0.001;
        this.scene.add(gridHelper);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    add(object) {
        this.scene.add(object);
    }

    render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.render();
    }
}
