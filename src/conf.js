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
    gridNoiseStrength = 0.5; // Noise strength for image mode (0 = no noise, 1 = full noise)
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
    mode = 0; // 0 = chaos, 1 = canvasser, 2 = protest, 3 = front gravity, 4 = front gravity with cylinder, 5 = sphere containment, 6 = color spheres, 9 = population segment pink
    cursorInteraction = false;
    hidePercentage = 0; // Percentage of particles to hide at edges (0-100)

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

    init(appInstance = null) {
        // Prevent multiple initializations
        if (this.gui) {
            console.log('Conf.init called multiple times, updating appInstance only. Old:', this.appInstance, 'New:', appInstance);
            this.appInstance = appInstance;
            return;
        }
        
        this.appInstance = appInstance;
        const gui = new Pane()
        gui.registerPlugin(EssentialsPlugin);

        const settings = gui.addFolder({
            title: "settings",
            expanded: false,
        });
        this.fpsGraph = settings.addBlade({
            view: 'fpsgraph',
            label: 'fps',
            rows: 2,
        });
        settings.addBinding(this, "particles", { min: 4096, max: this.maxParticles, step: 4096 }).on('change', () => { this.updateParams(); });
        settings.addBinding(this, "size", { min: 0.5, max: 2, step: 0.1 }).on('change', () => { this.updateParams(); });
        settings.addBinding(this, "run");
        settings.addBinding(this, "noise", { min: 0, max: 2, step: 0.01 });
        settings.addBinding(this, "gridNoiseStrength", { min: 0, max: 2, step: 0.01, label: "image noise" });
        settings.addBinding(this, "speed", { min: 0.1, max: 2, step: 0.1 });
        settings.addBinding(this, "density", { min: 0.4, max: 2, step: 0.1 }).on('change', () => { this.updateParams(); });
        settings.addBinding(this, "cursorInteraction", { label: "cursor interaction" });
        settings.addBinding(this, "hidePercentage", { min: 0, max: 100, step: 1, label: "% hidden" });
        
        // Mode selector as radio buttons - its own category
        const modeFolder = gui.addFolder({
            title: "mode",
            expanded: true,
        });
        
        const modeOptions = [
            {text: 'chaos', value: 0},
            {text: 'image – canvasser', value: 1},
            {text: 'image – protest', value: 2},
            {text: 'image – usa', value: 7},
            {text: 'front gravity', value: 3},
            {text: 'front gravity + cylinder', value: 4},
            {text: 'sphere containment', value: 5},
            {text: 'color spheres', value: 6},
            {text: 'polygon containment', value: 8},
            {text: 'population segment – pink', value: 9},
            {text: 'population segment – purple', value: 10},
            {text: 'population segment – cyan', value: 11},
            {text: 'population segment – teal', value: 12},
            {text: 'population segment – dark blue', value: 13},
            {text: 'population segment – dark purple', value: 14},
            {text: 'population segment – dark red', value: 15},
            {text: 'population segment – orange', value: 16},
            {text: 'population segment – light green', value: 17},
            {text: 'two color sphere', value: 18},
        ];
        
        // Store button references to update their colors
        const modeButtons = [];
        
        const updateButtonColors = () => {
            modeButtons.forEach((button, idx) => {
                const isActive = modeOptions[idx].value === this.mode;
                const buttonElement = button.controller.view.element;
                if (buttonElement) {
                    // Find the button element within the controller
                    const btn = buttonElement.querySelector('button') || buttonElement;
                    if (btn) {
                        if (isActive) {
                            btn.style.backgroundColor = '#4a9eff';
                            btn.style.color = '#ffffff';
                        } else {
                            btn.style.backgroundColor = '';
                            btn.style.color = '';
                        }
                    }
                }
            });
        };
        
        modeOptions.forEach((option) => {
            const btn = modeFolder.addButton({
                title: option.text,
            }).on('click', async () => {
                this.mode = option.value;
                updateButtonColors();
                
                // Always use the current appInstance (in case it was updated)
                const appInstance = this.appInstance;
                if (appInstance?.mlsMpmSim) {
                    console.log('Button clicked - calling toggleGridMode with mode:', option.value, 'appInstance:', appInstance);
                    // toggleGridMode is async, so await it
                    await appInstance.mlsMpmSim.toggleGridMode(option.value);
                } else {
                    console.error('Cannot toggle grid mode - appInstance or mlsMpmSim not available', {
                        hasAppInstance: !!appInstance,
                        hasMlsMpmSim: !!(appInstance?.mlsMpmSim)
                    });
                }
            });
            modeButtons.push(btn);
        });
        
        // Set initial button colors
        updateButtonColors();
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