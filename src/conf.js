import {Pane} from 'tweakpane';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';
import mobile from "is-mobile";
import * as THREE from "three/webgpu";

class Conf {
    gui = null;
    maxParticles = 8192 * 32; // Increased from 16 to 32 (262,144 particles)
    particles = 8192 * 10;

    bloom = true;

    run = true;
    noise = 1.0;
    speed = 1;
    stiffness = 3.;
    restDensity = 1.;
    density = 1;
    dynamicViscosity = 0.1;
    gravity = 0;
    gravitySensorReading = new THREE.Vector3();
    accelerometerReading = new THREE.Vector3();
    actualSize = 1;
    size = .5;

    points = false;
    mode = 0; // 0 = chaos, 1 = canvasser, 2 = protest, 3 = front gravity, 4 = front gravity with cylinder
    cursorInteraction = false;

    constructor(info) {
        if (mobile()) {
            this.maxParticles = 8192 * 16; // Increased from 8 to 16 (131,072 particles)
            this.particles = 4096;
        }
        this.updateParams();

    }

    updateParams() {
        const level = Math.max(this.particles / 8192,1);
        const size = 1.6/Math.pow(level, 1/3);
        this.actualSize = size * this.size;
        this.restDensity = 0.25 * level * this.density;
    }

    setupGravitySensor() {
        if (this.gravitySensor) { return; }
        this.gravitySensor = new GravitySensor({ frequency: 60 });
        this.gravitySensor.addEventListener("reading", (e) => {
            this.gravitySensorReading.copy(this.gravitySensor).divideScalar(50);
            this.gravitySensorReading.setY(this.gravitySensorReading.y * -1);
        });
        this.gravitySensor.start();
    }

    init() {
        const gui = new Pane()
        gui.registerPlugin(EssentialsPlugin);

        const stats = gui.addFolder({
            title: "stats",
            expanded: false,
        });
        this.fpsGraph = stats.addBlade({
            view: 'fpsgraph',
            label: 'fps',
            rows: 2,
        });

        const settings = gui.addFolder({
            title: "settings",
            expanded: false,
        });
        settings.addBinding(this, "particles", { min: 4096, max: this.maxParticles, step: 4096 }).on('change', () => { this.updateParams(); });
        settings.addBinding(this, "size", { min: 0.5, max: 2, step: 0.1 }).on('change', () => { this.updateParams(); });
        settings.addBinding(this, "run");
        settings.addBinding(this, "noise", { min: 0, max: 2, step: 0.01 });
        settings.addBinding(this, "speed", { min: 0.1, max: 2, step: 0.1 });
        settings.addBinding(this, "density", { min: 0.4, max: 2, step: 0.1 }).on('change', () => { this.updateParams(); });
        settings.addBinding(this, "cursorInteraction", { label: "cursor interaction" });
        settings.addBlade({
            view: 'list',
            label: 'mode',
            options: [
                {text: 'chaos', value: 0},
                {text: 'image – canvasser', value: 1},
                {text: 'image – protest', value: 2},
                {text: 'front gravity', value: 3},
                {text: 'front gravity + cylinder', value: 4},
            ],
            value: 0,
        }).on('change', (ev) => {
            this.mode = ev.value;
            if (window.app && window.app.mlsMpmSim) {
                window.app.mlsMpmSim.toggleGridMode(ev.value);
            }
        });
        //settings.addBinding(this, "points");

        /*settings.addBinding(this, "roughness", { min: 0.0, max: 1, step: 0.01 });
        settings.addBinding(this, "metalness", { min: 0.0, max: 1, step: 0.01 });*/

        this.gui = gui;
    }

    update() {
    }

    begin() {
        this.fpsGraph.begin();
    }
    end() {
        this.fpsGraph.end();
    }
}
export const conf = new Conf();