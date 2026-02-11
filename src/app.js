import * as THREE from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls"
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
import {Lights} from "./lights";

import { float, Fn, mrt, output, pass, vec3, vec4 } from "three/tsl";
import {conf} from "./conf";
import MlsMpmSimulator from "./mls-mpm/mlsMpmSimulator";
import ParticleRenderer from "./mls-mpm/particleRenderer";
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import PointRenderer from "./mls-mpm/pointRenderer.js";

const loadHdr = async (file) => {
    const texture = await new Promise(resolve => {
        new RGBELoader().load(file, result => {
            result.mapping = THREE.EquirectangularReflectionMapping;
            result.colo
            resolve(result);
        });
    });
    return texture;
}

class App {
    renderer = null;

    camera = null;

    scene = null;

    controls = null;

    lights = null;

    constructor(renderer) {
        this.renderer = renderer;
    }

    async init(progressCallback) {
        conf.init();

        this.camera = new THREE.PerspectiveCamera(20, window.innerWidth / window.innerHeight, 0.01, 5);
        this.camera.position.set(0, 0, -20);
        this.camera.updateProjectionMatrix()

        this.scene = new THREE.Scene();

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0,0.5,0.2);
        this.controls.enableDamping = true;
        this.controls.enablePan = false;
        this.controls.touches = {
            TWO: THREE.TOUCH.DOLLY_ROTATE,
        }
        this.controls.maxDistance = 3.0;
        this.controls.minPolarAngle = 0.2 * Math.PI;
        this.controls.maxPolarAngle = 0.8 * Math.PI;
        this.controls.minAzimuthAngle = 0.7 * Math.PI;
        this.controls.maxAzimuthAngle = 1.3 * Math.PI;

        await progressCallback(0.1)

        // Set plain white background
        this.scene.background = new THREE.Color(0xffffff);
        this.scene.environment = null;
        //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 0.66;

        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        await progressCallback(0.5)

        this.mlsMpmSim = new MlsMpmSimulator(this.renderer);
        await this.mlsMpmSim.init();
        this.particleRenderer = new ParticleRenderer(this.mlsMpmSim);
        this.scene.add(this.particleRenderer.object);
        this.pointRenderer = new PointRenderer(this.mlsMpmSim);
        this.scene.add(this.pointRenderer.object);
        
        // Expose app instance globally for conf access
        window.app = this;

        this.lights = new Lights();
        this.scene.add(this.lights.object);

        // Box container removed


        const scenePass = pass(this.scene, this.camera);
        scenePass.setMRT( mrt( {
            output,
            bloomIntensity: float( 0 ) // default bloom intensity
        } ) );
        const outputPass = scenePass.getTextureNode();
        const bloomIntensityPass = scenePass.getTextureNode( 'bloomIntensity' );
        const bloomPass = bloom( outputPass.mul( bloomIntensityPass ) );
        const postProcessing = new THREE.PostProcessing(this.renderer);
        postProcessing.outputColorTransform = false;
        //postProcessing.outputNode = vec4(outputPass.rgb, 1).add( vec4(bloomPass.mul(bloomIntensityPass.sign().oneMinus()).rgb, 0.0) ).renderOutput();
        //postProcessing.outputNode = outputPass.renderOutput();
        //(1-2b)*a*a + 2ba
        postProcessing.outputNode = Fn(() => {
            const a = outputPass.rgb.clamp(0,1).toVar();
            const b = bloomPass.rgb.clamp(0,1).mul(bloomIntensityPass.r.sign().oneMinus()).toVar();
            //return vec4(vec3(1).sub(b).sub(b).mul(a).mul(a).mul(0.0),1.0);
            //return b;
            //return a.div(b.oneMinus().max(0.0001)).clamp(0,1);
            return vec4(vec3(1).sub(b).sub(b).mul(a).mul(a).add(b.mul(a).mul(2)).clamp(0,1),1.0);
        })().renderOutput();

        this.postProcessing = postProcessing;
        this.bloomPass = bloomPass;
        this.bloomPass.threshold.value = 0.001;
        this.bloomPass.strength.value = 0.94;
        this.bloomPass.radius.value = 0.8;


        this.raycaster = new THREE.Raycaster();
        this.plane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0.2);
        this.renderer.domElement.addEventListener("pointermove", (event) => { this.onMouseMove(event); });

        await progressCallback(1.0, 100);
    }

    resize(width, height) {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    }

    onMouseMove(event) {
        const pointer = new THREE.Vector2();
        pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
        pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
        this.raycaster.setFromCamera(pointer, this.camera);
        const intersect = new THREE.Vector3();
        this.raycaster.ray.intersectPlane(this.plane, intersect);
        if (intersect) {
            this.mlsMpmSim.setMouseRay(this.raycaster.ray.origin, this.raycaster.ray.direction, intersect);
        }
    }


    async update(delta, elapsed) {
        conf.begin();

        this.particleRenderer.object.visible = !conf.points;
        this.pointRenderer.object.visible = conf.points;

        this.controls.update(delta);
        this.lights.update(elapsed);
        this.particleRenderer.update();
        this.pointRenderer.update();

        await this.mlsMpmSim.update(delta,elapsed);

        if (conf.bloom) {
            await this.postProcessing.renderAsync();
        } else {
            await this.renderer.renderAsync(this.scene, this.camera);
        }

        conf.end();
    }
}
export default App;
