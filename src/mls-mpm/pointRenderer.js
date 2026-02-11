import * as THREE from "three/webgpu";
import {Fn, vec3,instanceIndex} from "three/tsl";
import {conf} from "../conf";

class PointRenderer {
    mlsMpmSim = null;
    object = null;

    constructor(mlsMpmSim) {
        this.mlsMpmSim = mlsMpmSim;

        this.geometry = new THREE.InstancedBufferGeometry();
        const positionBuffer = new THREE.BufferAttribute(new Float32Array(3), 3, false);
        const material = new THREE.PointsNodeMaterial();
        this.geometry.setAttribute('position', positionBuffer);
        this.object = new THREE.Points(this.geometry, material);
        material.positionNode = Fn(() => {
            return this.mlsMpmSim.particleBuffer.element(instanceIndex).get('position').mul(vec3(1,1,0.4));
        })();

        this.object.frustumCulled = false;

        const s = (1/64);
        this.object.position.set(-64.0*s,0,0); // Updated for 2x wider container
        this.object.scale.set(s,s,s);
        this.object.castShadow = true;
        this.object.receiveShadow = true;
    }

    update() {
        const { particles } = conf;
        this.geometry.instanceCount = particles;
    }
}
export default PointRenderer;