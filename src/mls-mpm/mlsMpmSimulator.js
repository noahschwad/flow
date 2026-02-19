import * as THREE from "three/webgpu";
import {
    array,
    Fn,
    If,
    instancedArray,
    instanceIndex,
    Return,
    uniform,
    int,
    float,
    Loop,
    vec2,
    vec3,
    vec4,
    atomicAdd,
    uint,
    max,
    pow,
    mat3,
    clamp,
    time,
    cross, mix, mx_hsvtorgb, select, ivec3
} from "three/tsl";
import {triNoise3Dvec} from "../common/noise";
import {conf} from "../conf";
import {StructuredArray} from "./structuredArray.js";
import {hsvtorgb} from "../common/hsv.js";
import canvasserImage from "../assets/canvasser.png";
import protestImage from "../assets/protest.png";
import usaImage from "../assets/usa.jpg";

class mlsMpmSimulator {
    static COLOR_SPHERES_BOOST_DURATION = 1.4; // Duration of boost period in seconds
    
    renderer = null;
    particleRenderer = null; // Reference to particleRenderer for material updates
    numParticles = 0;
    gridSize = new THREE.Vector3(0,0,0);
    gridCellSize = new THREE.Vector3(0,0,0);
    uniforms = {};
    kernels = {};
    fixedPointMultiplier = 1e7;
    mousePos = new THREE.Vector3();
    mousePosArray = [];
    gridMode = false;
    currentImageMode = null; // Track which image is currently loaded
    imageData = null;
    gridTargetPositions = [];
    gridTargetColors = [];
    gridWidth = 0;
    gridHeight = 0;
    gridStartX = 0;
    gridStartY = 0;
    gridSpacingX = 0;
    gridSpacingY = 0;
    colorSpherePositions = []; // Array to store sphere positions for each palette color
    paletteRGB = []; // Store palette colors for color matching
    colorSpheresModeStartTime = 0; // Track when color spheres mode was activated
    polygonVertices = []; // Store polygon vertices for V containment mode
    populationSegmentPinkModeStartTime = 0; // Track when population segment pink mode was activated
    populationSegmentPurpleModeStartTime = 0; // Track when population segment purple mode was activated
    populationSegmentCyanModeStartTime = 0; // Track when population segment cyan mode was activated
    populationSegmentTealModeStartTime = 0; // Track when population segment teal mode was activated
    populationSegmentDarkBlueModeStartTime = 0;
    populationSegmentDarkPurpleModeStartTime = 0;
    populationSegmentDarkRedModeStartTime = 0;
    populationSegmentOrangeModeStartTime = 0;
    populationSegmentLightGreenModeStartTime = 0;

    constructor(renderer) {
        this.renderer = renderer;
    }
    async init() {
        const {maxParticles} = conf;
        this.gridSize.set(128,64,64); // 2x wider in x dimension

        const particleStruct =  {
            position: { type: 'vec3' },
            density: { type: 'float' },
            velocity: { type: 'vec3' },
            mass: { type: 'float' },
            C: { type: 'mat3' },
            direction: { type: 'vec3' },
            color: { type: 'vec3' },
        };
        this.particleBuffer = new StructuredArray(particleStruct, maxParticles, "particleData");

        // Color palette
        const palette = [
            "#FF56B7",
            "#BC96F9",
            "#C5F0F2",
            "#007487",
            "#007487",
            "#153943",
            "#232356",
            "#50121A",
            "#FF6425",
            "#D6F499",
            "#5CCAD8",
            "#5CCAD8",
            "#C5F0F2",
        ];

        // Convert hex colors to RGB vec3
        const paletteRGB = palette.map(hex => {
            const r = parseInt(hex.slice(1, 3), 16) / 255;
            const g = parseInt(hex.slice(3, 5), 16) / 255;
            const b = parseInt(hex.slice(5, 7), 16) / 255;
            return new THREE.Vector3(r, g, b);
        });

        const vec = new THREE.Vector3();
        for (let i = 0; i < maxParticles; i++) {
            let dist = 2;
            while (dist > 1) {
                vec.set(Math.random(),Math.random(),Math.random()).multiplyScalar(2.0).subScalar(1.0);
                dist = vec.length();
                vec.multiplyScalar(0.8).addScalar(1.0).divideScalar(2.0).multiply(this.gridSize);
            }
            const mass = 1.0 - Math.random() * 0.002;
            // Assign color from palette - cycle through palette to ensure all colors are used
            const colorIndex = i % paletteRGB.length;
            const assignedColor = paletteRGB[colorIndex];
            this.particleBuffer.set(i, "position", vec);
            this.particleBuffer.set(i, "mass", mass);
            this.particleBuffer.set(i, "color", assignedColor);
        }

        const cellCount = this.gridSize.x * this.gridSize.y * this.gridSize.z;
        const cellStruct ={
            x: { type: 'int', atomic: true },
            y: { type: 'int', atomic: true },
            z: { type: 'int', atomic: true },
            mass: { type: 'int', atomic: true },
        };
        this.cellBuffer = new StructuredArray(cellStruct, cellCount, "cellData");
        this.cellBufferF = instancedArray(cellCount, 'vec4').label('cellDataF');

        this.uniforms.gravityType = uniform(0, "uint");
        this.uniforms.gravity = uniform(new THREE.Vector3());
        this.uniforms.stiffness = uniform(0);
        this.uniforms.restDensity = uniform(0);
        this.uniforms.dynamicViscosity = uniform(0);
        this.uniforms.noise = uniform(0);

        this.uniforms.gridSize = uniform(this.gridSize, "ivec3");
        this.uniforms.gridCellSize = uniform(this.gridCellSize);
        this.uniforms.dt = uniform(0.1);
        this.uniforms.numParticles = uniform(0, "uint");
        this.uniforms.gridMode = uniform(0, "uint");
        this.uniforms.frontGravityMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.cylinderMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.cylinderRadius = uniform(44.0); // Radius of the repulsive cylinder
        this.uniforms.cylinderCenterX = uniform(64.0); // Center X of cylinder (center of box width)
        this.uniforms.cylinderCenterY = uniform(32.0); // Center Y of cylinder (center of box height)
        this.uniforms.sphereContainmentMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.polygonContainmentMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.colorSpheresMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.twoColorSphereMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.populationSegmentPinkMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.populationSegmentPurpleMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.populationSegmentCyanMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.populationSegmentTealMode = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.populationSegmentDarkBlueMode = uniform(0, "uint");
        this.uniforms.populationSegmentDarkPurpleMode = uniform(0, "uint");
        this.uniforms.populationSegmentDarkRedMode = uniform(0, "uint");
        this.uniforms.populationSegmentOrangeMode = uniform(0, "uint");
        this.uniforms.populationSegmentLightGreenMode = uniform(0, "uint");
        // Polygon vertices - support up to 16 vertices (enough for the V shape)
        this.uniforms.polygonVertexCount = uniform(0, "uint");
        this.uniforms.polygonVertices = [];
        for (let i = 0; i < 16; i++) {
            this.uniforms.polygonVertices.push(uniform(new THREE.Vector2()));
        }
        this.uniforms.colorSpheresBoostTime = uniform(0); // Time remaining for boost (0 = no boost)
        this.uniforms.colorSpheresBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION); // Duration of boost period
        this.uniforms.populationSegmentPinkBoostTime = uniform(0); // Time remaining for boost (0 = no boost)
        this.uniforms.populationSegmentPinkBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION); // Duration of boost period
        this.uniforms.populationSegmentPurpleBoostTime = uniform(0);
        this.uniforms.populationSegmentPurpleBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentCyanBoostTime = uniform(0);
        this.uniforms.populationSegmentCyanBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentTealBoostTime = uniform(0);
        this.uniforms.populationSegmentTealBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentDarkBlueBoostTime = uniform(0);
        this.uniforms.populationSegmentDarkBlueBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentDarkPurpleBoostTime = uniform(0);
        this.uniforms.populationSegmentDarkPurpleBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentDarkRedBoostTime = uniform(0);
        this.uniforms.populationSegmentDarkRedBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentOrangeBoostTime = uniform(0);
        this.uniforms.populationSegmentOrangeBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.populationSegmentLightGreenBoostTime = uniform(0);
        this.uniforms.populationSegmentLightGreenBoostDuration = uniform(mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION);
        this.uniforms.pinkSphereRadius = uniform(7.0); // Radius of pink particle sphere
        this.uniforms.pinkSphereCenter = uniform(new THREE.Vector3()); // Center position of pink sphere
        this.uniforms.purpleSphereRadius = uniform(7.0);
        this.uniforms.purpleSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.cyanSphereRadius = uniform(7.0);
        this.uniforms.cyanSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.tealSphereRadius = uniform(7.0);
        this.uniforms.tealSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.darkBlueSphereRadius = uniform(7.0);
        this.uniforms.darkBlueSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.darkPurpleSphereRadius = uniform(7.0);
        this.uniforms.darkPurpleSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.darkRedSphereRadius = uniform(7.0);
        this.uniforms.darkRedSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.orangeSphereRadius = uniform(7.0);
        this.uniforms.orangeSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.lightGreenSphereRadius = uniform(7.0);
        this.uniforms.lightGreenSphereCenter = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereRadius = uniform(17.0); // Radius of negative space sphere (1.5x pink sphere radius)
        this.uniforms.negativeSpaceSphereCenter = uniform(new THREE.Vector3()); // Center position of negative space sphere
        this.uniforms.negativeSpaceSphereCenterPurple = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterCyan = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterTeal = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterDarkBlue = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterDarkPurple = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterDarkRed = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterOrange = uniform(new THREE.Vector3());
        this.uniforms.negativeSpaceSphereCenterLightGreen = uniform(new THREE.Vector3());
        this.uniforms.populationSegmentPurpleMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentCyanMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentTealMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentDarkBlueMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentDarkPurpleMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentDarkRedMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentOrangeMainSphereRadius = uniform(18.0);
        this.uniforms.populationSegmentLightGreenMainSphereRadius = uniform(18.0);
        this.uniforms.sphereRadius = uniform(25.0); // Radius of containment sphere (half of box width = 128/2)
        this.uniforms.populationSegmentPinkMainSphereRadius = uniform(18.0); // Smaller radius for main sphere in population segment pink mode
        this.uniforms.colorSphereRadius = uniform(7.0); // Radius of individual color spheres
        this.uniforms.sphereCenterX = uniform(64.0); // Center X of sphere (center of box width)
        this.uniforms.sphereCenterY = uniform(32.0); // Center Y of sphere (center of box height)
        this.uniforms.sphereCenterZ = uniform(32.0); // Center Z of sphere (center of box depth)
        // Color sphere positions - stored as separate uniforms for each unique color (10 unique colors)
        this.uniforms.colorSpherePositions = []; // Will be array of vec3 uniforms
        this.uniforms.paletteColors = []; // Will be array of vec3 uniforms for matching
        for (let i = 0; i < 13; i++) {
            this.uniforms.colorSpherePositions.push(uniform(new THREE.Vector3()));
            this.uniforms.paletteColors.push(uniform(new THREE.Vector3()));
        }
        this.uniforms.gridWidth = uniform(0, "uint");
        this.uniforms.gridHeight = uniform(0, "uint");
        this.uniforms.gridStartX = uniform(0);
        this.uniforms.gridStartY = uniform(0);
        this.uniforms.gridSpacingX = uniform(0);
        this.uniforms.gridSpacingY = uniform(0);
        this.uniforms.gridZ = uniform(32);
        this.uniforms.gridRandomOffset = uniform(.5); // Random offset amount (as fraction of spacing)
        this.uniforms.gridNoiseStrength = uniform(3.5); // Noise strength for image mode (0 = no noise, 1 = full noise)

        this.uniforms.mouseRayDirection = uniform(new THREE.Vector3());
        this.uniforms.mouseRayOrigin = uniform(new THREE.Vector3());
        this.uniforms.mouseForce = uniform(new THREE.Vector3());
        this.uniforms.cursorInteraction = uniform(0, "uint"); // 0 = off, 1 = on
        this.uniforms.hidePercentage = uniform(0.0); // Percentage of particles to hide at edges (0-100)

        this.kernels.clearGrid = Fn(() => {
            this.cellBuffer.setAtomic("x", false);
            this.cellBuffer.setAtomic("y", false);
            this.cellBuffer.setAtomic("z", false);
            this.cellBuffer.setAtomic("mass", false);

            If(instanceIndex.greaterThanEqual(uint(cellCount)), () => {
                Return();
            });

            this.cellBuffer.element(instanceIndex).get('x').assign(0);
            this.cellBuffer.element(instanceIndex).get('y').assign(0);
            this.cellBuffer.element(instanceIndex).get('z').assign(0);
            this.cellBuffer.element(instanceIndex).get('mass').assign(0);
            this.cellBufferF.element(instanceIndex).assign(0);
        })().compute(cellCount);

        const encodeFixedPoint = (f32) => {
            return int(f32.mul(this.fixedPointMultiplier));
        }
        const decodeFixedPoint = (i32) => {
            return float(i32).div(this.fixedPointMultiplier);
        }

        const getCellPtr = (ipos) => {
            const gridSize = this.uniforms.gridSize;
            const cellPtr = int(ipos.x).mul(gridSize.y).mul(gridSize.z).add(int(ipos.y).mul(gridSize.z)).add(int(ipos.z)).toConst();
            return cellPtr;
        };
        const getCell = (ipos) => {
            return this.cellBuffer.element(getCellPtr(ipos));
        };

        this.kernels.p2g1 = Fn(() => {
            this.cellBuffer.setAtomic("x", true);
            this.cellBuffer.setAtomic("y", true);
            this.cellBuffer.setAtomic("z", true);
            this.cellBuffer.setAtomic("mass", true);

            If(instanceIndex.greaterThanEqual(uint(this.uniforms.numParticles)), () => {
                Return();
            });
            const particlePosition = this.particleBuffer.element(instanceIndex).get('position').xyz.toConst("particlePosition");
            const particleVelocity = this.particleBuffer.element(instanceIndex).get('velocity').xyz.toConst("particleVelocity");

            const cellIndex =  ivec3(particlePosition).sub(1).toConst("cellIndex");
            const cellDiff = particlePosition.fract().sub(0.5).toConst("cellDiff");
            const w0 = float(0.5).mul(float(0.5).sub(cellDiff)).mul(float(0.5).sub(cellDiff));
            const w1 = float(0.75).sub(cellDiff.mul(cellDiff));
            const w2 = float(0.5).mul(float(0.5).add(cellDiff)).mul(float(0.5).add(cellDiff));
            const weights = array([w0,w1,w2]).toConst("weights");

            const C = this.particleBuffer.element(instanceIndex).get('C').toConst();
            Loop({ start: 0, end: 3, type: 'int', name: 'gx', condition: '<' }, ({gx}) => {
                Loop({ start: 0, end: 3, type: 'int', name: 'gy', condition: '<' }, ({gy}) => {
                    Loop({ start: 0, end: 3, type: 'int', name: 'gz', condition: '<' }, ({gz}) => {
                        const weight = weights.element(gx).x.mul(weights.element(gy).y).mul(weights.element(gz).z);
                        const cellX = cellIndex.add(ivec3(gx,gy,gz)).toConst();
                        const cellDist = vec3(cellX).add(0.5).sub(particlePosition).toConst("cellDist");
                        const Q = C.mul(cellDist);

                        const massContrib = weight; // assuming particle mass = 1.0
                        const velContrib = massContrib.mul(particleVelocity.add(Q)).toConst("velContrib");
                        const cell = getCell(cellX);
                        atomicAdd(cell.get('x'), encodeFixedPoint(velContrib.x));
                        atomicAdd(cell.get('y'), encodeFixedPoint(velContrib.y));
                        atomicAdd(cell.get('z'), encodeFixedPoint(velContrib.z));
                        atomicAdd(cell.get('mass'), encodeFixedPoint(massContrib));
                    });
                });
            });
        })().compute(1);


        this.kernels.p2g2 = Fn(() => {
            this.cellBuffer.setAtomic("x", true);
            this.cellBuffer.setAtomic("y", true);
            this.cellBuffer.setAtomic("z", true);
            this.cellBuffer.setAtomic("mass", false);

            If(instanceIndex.greaterThanEqual(uint(this.uniforms.numParticles)), () => {
                Return();
            });
            const particlePosition = this.particleBuffer.element(instanceIndex).get('position').xyz.toConst("particlePosition");

            const cellIndex =  ivec3(particlePosition).sub(1).toConst("cellIndex");
            const cellDiff = particlePosition.fract().sub(0.5).toConst("cellDiff");
            const w0 = float(0.5).mul(float(0.5).sub(cellDiff)).mul(float(0.5).sub(cellDiff));
            const w1 = float(0.75).sub(cellDiff.mul(cellDiff));
            const w2 = float(0.5).mul(float(0.5).add(cellDiff)).mul(float(0.5).add(cellDiff));
            const weights = array([w0,w1,w2]).toConst("weights");

            const density = float(0).toVar("density");
            Loop({ start: 0, end: 3, type: 'int', name: 'gx', condition: '<' }, ({gx}) => {
                Loop({ start: 0, end: 3, type: 'int', name: 'gy', condition: '<' }, ({gy}) => {
                    Loop({ start: 0, end: 3, type: 'int', name: 'gz', condition: '<' }, ({gz}) => {
                        const weight = weights.element(gx).x.mul(weights.element(gy).y).mul(weights.element(gz).z);
                        const cellX = cellIndex.add(ivec3(gx,gy,gz)).toConst();
                        const cell = getCell(cellX);
                        density.addAssign(decodeFixedPoint(cell.get('mass')).mul(weight));
                    });
                });
            });
            const densityStore = this.particleBuffer.element(instanceIndex).get('density');
            densityStore.assign(mix(densityStore, density, 0.05));

            const volume = float(1).div(density);
            const pressure = max(0.0, pow(density.div(this.uniforms.restDensity), 5.0).sub(1).mul(this.uniforms.stiffness)).toConst('pressure');
            const stress = mat3(pressure.negate(), 0, 0, 0, pressure.negate(), 0, 0, 0, pressure.negate()).toVar('stress');
            const dudv = this.particleBuffer.element(instanceIndex).get('C').toConst('C');

            const strain = dudv.add(dudv.transpose());
            stress.addAssign(strain.mul(this.uniforms.dynamicViscosity));
            const eq16Term0 = volume.mul(-4).mul(stress).mul(this.uniforms.dt);

            Loop({ start: 0, end: 3, type: 'int', name: 'gx', condition: '<' }, ({gx}) => {
                Loop({ start: 0, end: 3, type: 'int', name: 'gy', condition: '<' }, ({gy}) => {
                    Loop({ start: 0, end: 3, type: 'int', name: 'gz', condition: '<' }, ({gz}) => {
                        const weight = weights.element(gx).x.mul(weights.element(gy).y).mul(weights.element(gz).z);
                        const cellX = cellIndex.add(ivec3(gx,gy,gz)).toConst();
                        const cellDist = vec3(cellX).add(0.5).sub(particlePosition).toConst("cellDist");
                        const cell= getCell(cellX);

                        const momentum = eq16Term0.mul(weight).mul(cellDist).toConst("momentum");
                        atomicAdd(cell.get('x'), encodeFixedPoint(momentum.x));
                        atomicAdd(cell.get('y'), encodeFixedPoint(momentum.y));
                        atomicAdd(cell.get('z'), encodeFixedPoint(momentum.z));
                    });
                });
            });
        })().compute(1);


        this.kernels.updateGrid = Fn(() => {
            this.cellBuffer.setAtomic("x", false);
            this.cellBuffer.setAtomic("y", false);
            this.cellBuffer.setAtomic("z", false);
            this.cellBuffer.setAtomic("mass", false);

            If(instanceIndex.greaterThanEqual(uint(cellCount)), () => {
                Return();
            });
            const cell = this.cellBuffer.element(instanceIndex).toConst("cell");

            const mass = decodeFixedPoint(cell.get('mass')).toConst();
            If(mass.lessThanEqual(0), () => { Return(); });

            const vx = decodeFixedPoint(cell.get('x')).div(mass).toVar();
            const vy = decodeFixedPoint(cell.get('y')).div(mass).toVar();
            const vz = decodeFixedPoint(cell.get('z')).div(mass).toVar();

            const x = int(instanceIndex).div(this.uniforms.gridSize.z).div(this.uniforms.gridSize.y);
            const y = int(instanceIndex).div(this.uniforms.gridSize.z).mod(this.uniforms.gridSize.y);
            const z = int(instanceIndex).mod(this.uniforms.gridSize.z);


            If(x.lessThan(int(2)).or(x.greaterThan(this.uniforms.gridSize.x.sub(int(2)))), () => {
                vx.assign(0);
            });
            If(y.lessThan(int(2)).or(y.greaterThan(this.uniforms.gridSize.y.sub(int(2)))), () => {
                vy.assign(0);
            });
            If(z.lessThan(int(2)).or(z.greaterThan(this.uniforms.gridSize.z.sub(int(2)))), () => {
                vz.assign(0);
            });

            this.cellBufferF.element(instanceIndex).assign(vec4(vx,vy,vz,mass));
        })().compute(cellCount);

        this.kernels.g2p = Fn(() => {
            If(instanceIndex.greaterThanEqual(uint(this.uniforms.numParticles)), () => {
                Return();
            });
            const particleMass = this.particleBuffer.element(instanceIndex).get('mass').toConst("particleMass");
            const particleDensity = this.particleBuffer.element(instanceIndex).get('density').toConst("particleDensity");
            const particlePosition = this.particleBuffer.element(instanceIndex).get('position').xyz.toVar("particlePosition");
            const particleVelocity = vec3(0).toVar();
            
            // Calculate if particle should be hidden (early, so we can skip mode forces for hidden particles)
            const particleIndexPercent = float(instanceIndex).div(float(this.uniforms.numParticles)).mul(100.0).toConst();
            const shouldHide = this.uniforms.hidePercentage.greaterThan(0.0).and(particleIndexPercent.lessThan(this.uniforms.hidePercentage)).toConst();
            
            // Grid mode: apply force towards target grid position (skip for hidden particles)
            If(this.uniforms.gridMode.equal(uint(1)).and(shouldHide.not()), () => {
                const gridWidthInt = int(this.uniforms.gridWidth);
                const gridX = int(instanceIndex).mod(gridWidthInt);
                const gridY = int(instanceIndex).div(gridWidthInt);
                
                // Generate pseudo-random offsets based on particle index for consistent randomness
                // Use a simple hash function to create random values from instanceIndex
                const hash = float(instanceIndex).mul(12.9898).add(float(gridX).mul(78.233)).add(float(gridY).mul(37.719));
                const randomX = hash.fract().mul(2.0).sub(1.0); // -1 to 1
                const randomY = hash.mul(43758.5453).fract().mul(2.0).sub(1.0); // -1 to 1
                
                // Apply random offset scaled by spacing
                const offsetX = randomX.mul(this.uniforms.gridSpacingX).mul(this.uniforms.gridRandomOffset);
                const offsetY = randomY.mul(this.uniforms.gridSpacingY).mul(this.uniforms.gridRandomOffset);
                
                const targetX = float(this.uniforms.gridStartX).add(float(gridX).mul(this.uniforms.gridSpacingX)).add(offsetX);
                const targetY = float(this.uniforms.gridStartY).add(float(gridY).mul(this.uniforms.gridSpacingY)).add(offsetY);
                let targetPos = vec3(targetX, targetY, this.uniforms.gridZ).toVar();
                
                // Apply dynamic Perlin noise to position in image mode
                // Use particle position and time to generate smooth, dynamic noise
                const noiseScale = float(0.92).toConst(); // Scale for noise sampling
                const noiseSpeed = float(1.15).toConst(); // Speed of noise animation
                const noiseOffset = triNoise3Dvec(particlePosition.mul(noiseScale), noiseSpeed, time).mul(this.uniforms.gridNoiseStrength).mul(float(2.0)).toConst();
                targetPos.addAssign(noiseOffset);
                
                // Apply additional Perlin noise specifically to z position
                // Use a different noise scale and offset for z to create independent variation
                const zNoiseScale = float(0.85).toConst(); // Scale for z noise sampling
                const zNoiseSpeed = float(0.02).toConst(); // Speed of z noise animation
                const zNoiseOffset = triNoise3Dvec(particlePosition.mul(zNoiseScale).add(vec3(100.0, 200.0, 300.0)), zNoiseSpeed, time).z.mul(this.uniforms.gridNoiseStrength).mul(float(2.0)).toConst();
                targetPos.z.addAssign(zNoiseOffset);
                
                const toTarget = targetPos.sub(particlePosition);
                const dist = toTarget.length();
                // Apply force proportional to distance - no max speed limit in grid mode
                const forceStrength = dist.mul(10.0); // Increased multiplier, no clamp for unlimited speed
                particleVelocity.addAssign(toTarget.normalize().mul(forceStrength).mul(this.uniforms.dt));
            }).Else(() => {
                // Normal fluid simulation forces or front gravity (skip for hidden particles)
                If(this.uniforms.frontGravityMode.equal(uint(1)).and(shouldHide.not()), () => {
                    // Front gravity mode: pull toward front (negative Z direction)
                    const frontGravityDir = vec3(0, 0, -1).toConst(); // Negative Z is front
                    particleVelocity.addAssign(frontGravityDir.mul(0.1).mul(this.uniforms.dt));
                }).Else(() => {
                    // Normal center gravity
                    const pn = particlePosition.div(vec3(this.uniforms.gridSize.sub(1))).sub(0.5).normalize().toConst();
                    particleVelocity.subAssign(pn.mul(0.1).mul(this.uniforms.dt));
                });
            });
            
            // Apply cylinder repulsion force when in cylinder mode (skip for hidden particles)
            If(this.uniforms.cylinderMode.equal(uint(1)).and(shouldHide.not()), () => {
                // Calculate distance from particle to cylinder axis (in XY plane)
                const cylinderAxis = vec3(this.uniforms.cylinderCenterX, this.uniforms.cylinderCenterY, particlePosition.z).toConst();
                const toCylinderAxis = particlePosition.sub(cylinderAxis).toConst();
                const distToAxis = vec3(toCylinderAxis.x, toCylinderAxis.y, float(0)).length().toConst(); // Distance in XY plane
                
                // If particle is inside cylinder radius, push it away
                If(distToAxis.lessThan(this.uniforms.cylinderRadius), () => {
                    // Calculate repulsion direction (away from cylinder axis in XY plane)
                    const repulsionDir = vec3(toCylinderAxis.x, toCylinderAxis.y, float(0)).normalize().toConst();
                    // Strength increases as particle gets closer to center
                    const repulsionStrength = float(1.0).sub(distToAxis.div(this.uniforms.cylinderRadius)).mul(0.5).toConst();
                    particleVelocity.addAssign(repulsionDir.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply sphere containment force when in sphere containment mode (skip for hidden particles)
            If(this.uniforms.sphereContainmentMode.equal(uint(1)).and(shouldHide.not()), () => {
                // Calculate distance from particle to sphere center
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                
                // If particle is outside sphere radius, push it back in
                If(distToCenter.greaterThan(this.uniforms.sphereRadius), () => {
                    // Calculate direction toward center (normalized)
                    const towardCenter = toCenter.normalize().negate().toConst();
                    // Strength increases with distance outside sphere
                    const excessDistance = distToCenter.sub(this.uniforms.sphereRadius).toConst();
                    const containmentStrength = excessDistance.mul(0.3).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply sphere containment force when in two color sphere mode (skip for hidden particles)
            If(this.uniforms.twoColorSphereMode.equal(uint(1)).and(shouldHide.not()), () => {
                // Calculate distance from particle to sphere center
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                
                // If particle is outside sphere radius, push it back in
                If(distToCenter.greaterThan(this.uniforms.sphereRadius), () => {
                    // Calculate direction toward center (normalized)
                    const towardCenter = toCenter.normalize().negate().toConst();
                    // Strength increases with distance outside sphere
                    const excessDistance = distToCenter.sub(this.uniforms.sphereRadius).toConst();
                    const containmentStrength = excessDistance.mul(0.3).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply polygon containment force when in polygon containment mode (skip for hidden particles)
            If(this.uniforms.polygonContainmentMode.equal(uint(1)).and(shouldHide.not()), () => {
                // Point-in-polygon test using ray casting algorithm
                // Project particle position to XY plane
                const p = vec2(particlePosition.x, particlePosition.y).toConst();
                let inside = int(0).toVar(); // 0 = outside, 1 = inside
                
                // Cast a ray from point to right (positive X) and count edge crossings
                const vertexCount = int(this.uniforms.polygonVertexCount);
                Loop({ start: 0, end: vertexCount, type: 'int', name: 'i', condition: '<' }, ({i}) => {
                    const j = select(
                        i.equal(vertexCount.sub(1)), int(0),
                        i.add(1)
                    ).toConst(); // Next vertex (wrapping around)
                    
                    // Get vertices using select (since we can't index arrays directly)
                    const v0 = select(
                        i.equal(0), this.uniforms.polygonVertices[0],
                        select(i.equal(1), this.uniforms.polygonVertices[1],
                        select(i.equal(2), this.uniforms.polygonVertices[2],
                        select(i.equal(3), this.uniforms.polygonVertices[3],
                        select(i.equal(4), this.uniforms.polygonVertices[4],
                        select(i.equal(5), this.uniforms.polygonVertices[5],
                        select(i.equal(6), this.uniforms.polygonVertices[6],
                        select(i.equal(7), this.uniforms.polygonVertices[7],
                        select(i.equal(8), this.uniforms.polygonVertices[8],
                        select(i.equal(9), this.uniforms.polygonVertices[9],
                        select(i.equal(10), this.uniforms.polygonVertices[10],
                        select(i.equal(11), this.uniforms.polygonVertices[11],
                        select(i.equal(12), this.uniforms.polygonVertices[12],
                        select(i.equal(13), this.uniforms.polygonVertices[13],
                        select(i.equal(14), this.uniforms.polygonVertices[14],
                        this.uniforms.polygonVertices[15]))))))))))))))).toConst();
                    
                    const v1 = select(
                        j.equal(0), this.uniforms.polygonVertices[0],
                        select(j.equal(1), this.uniforms.polygonVertices[1],
                        select(j.equal(2), this.uniforms.polygonVertices[2],
                        select(j.equal(3), this.uniforms.polygonVertices[3],
                        select(j.equal(4), this.uniforms.polygonVertices[4],
                        select(j.equal(5), this.uniforms.polygonVertices[5],
                        select(j.equal(6), this.uniforms.polygonVertices[6],
                        select(j.equal(7), this.uniforms.polygonVertices[7],
                        select(j.equal(8), this.uniforms.polygonVertices[8],
                        select(j.equal(9), this.uniforms.polygonVertices[9],
                        select(j.equal(10), this.uniforms.polygonVertices[10],
                        select(j.equal(11), this.uniforms.polygonVertices[11],
                        select(j.equal(12), this.uniforms.polygonVertices[12],
                        select(j.equal(13), this.uniforms.polygonVertices[13],
                        select(j.equal(14), this.uniforms.polygonVertices[14],
                        this.uniforms.polygonVertices[15]))))))))))))))).toConst();
                    
                    // Check if ray crosses this edge
                    // Ray is horizontal from p.y to infinity in +X direction
                    const y0 = v0.y;
                    const y1 = v1.y;
                    const x0 = v0.x;
                    const x1 = v1.x;
                    
                    // Edge crosses ray if: one vertex is above p.y and one is below/on
                    const above0 = y0.greaterThan(p.y).toConst();
                    const above1 = y1.greaterThan(p.y).toConst();
                    const crosses = above0.notEqual(above1).toConst();
                    
                    If(crosses, () => {
                        // Calculate X intersection of edge with horizontal line at p.y
                        const t = p.y.sub(y0).div(y1.sub(y0)).toConst();
                        const xIntersect = x0.add(x1.sub(x0).mul(t)).toConst();
                        
                        // Ray crosses if intersection is to the right of point
                        If(xIntersect.greaterThan(p.x), () => {
                            inside.assign(inside.add(1).mod(2)); // Toggle inside/outside
                        });
                    });
                });
                
                // If particle is outside polygon, push it toward nearest edge
                If(inside.equal(0), () => {
                    // Find nearest edge and push toward it
                    let minDist = float(9999.0).toVar();
                    let nearestPoint = vec2(0).toVar();
                    
                    Loop({ start: 0, end: vertexCount, type: 'int', name: 'i', condition: '<' }, ({i}) => {
                        const j = select(
                            i.equal(vertexCount.sub(1)), int(0),
                            i.add(1)
                        ).toConst();
                        
                        const v0 = select(
                            i.equal(0), this.uniforms.polygonVertices[0],
                            select(i.equal(1), this.uniforms.polygonVertices[1],
                            select(i.equal(2), this.uniforms.polygonVertices[2],
                            select(i.equal(3), this.uniforms.polygonVertices[3],
                            select(i.equal(4), this.uniforms.polygonVertices[4],
                            select(i.equal(5), this.uniforms.polygonVertices[5],
                            select(i.equal(6), this.uniforms.polygonVertices[6],
                            select(i.equal(7), this.uniforms.polygonVertices[7],
                            select(i.equal(8), this.uniforms.polygonVertices[8],
                            select(i.equal(9), this.uniforms.polygonVertices[9],
                            select(i.equal(10), this.uniforms.polygonVertices[10],
                            select(i.equal(11), this.uniforms.polygonVertices[11],
                            select(i.equal(12), this.uniforms.polygonVertices[12],
                            select(i.equal(13), this.uniforms.polygonVertices[13],
                            select(i.equal(14), this.uniforms.polygonVertices[14],
                            this.uniforms.polygonVertices[15]))))))))))))))).toConst();
                        
                        const v1 = select(
                            j.equal(0), this.uniforms.polygonVertices[0],
                            select(j.equal(1), this.uniforms.polygonVertices[1],
                            select(j.equal(2), this.uniforms.polygonVertices[2],
                            select(j.equal(3), this.uniforms.polygonVertices[3],
                            select(j.equal(4), this.uniforms.polygonVertices[4],
                            select(j.equal(5), this.uniforms.polygonVertices[5],
                            select(j.equal(6), this.uniforms.polygonVertices[6],
                            select(j.equal(7), this.uniforms.polygonVertices[7],
                            select(j.equal(8), this.uniforms.polygonVertices[8],
                            select(j.equal(9), this.uniforms.polygonVertices[9],
                            select(j.equal(10), this.uniforms.polygonVertices[10],
                            select(j.equal(11), this.uniforms.polygonVertices[11],
                            select(j.equal(12), this.uniforms.polygonVertices[12],
                            select(j.equal(13), this.uniforms.polygonVertices[13],
                            select(j.equal(14), this.uniforms.polygonVertices[14],
                            this.uniforms.polygonVertices[15]))))))))))))))).toConst();
                        
                        // Find closest point on edge segment
                        const edge = v1.sub(v0).toConst();
                        const toP = p.sub(v0).toConst();
                        const edgeLenSq = edge.length().pow(2).toConst();
                        const t = select(
                            edgeLenSq.lessThan(0.0001), float(0.0),
                            clamp(toP.dot(edge).div(edgeLenSq), float(0.0), float(1.0))
                        ).toConst();
                        const closestOnEdge = v0.add(edge.mul(t)).toConst();
                        const dist = p.sub(closestOnEdge).length().toConst();
                        
                        If(dist.lessThan(minDist), () => {
                            minDist.assign(dist);
                            nearestPoint.assign(closestOnEdge);
                        });
                    });
                    
                    // Push particle toward nearest point on polygon edge
                    const toNearest = vec3(nearestPoint.x, nearestPoint.y, particlePosition.z).sub(particlePosition).toConst();
                    const distToNearest = toNearest.length().toConst();
                    const containmentStrength = distToNearest.mul(0.3).toConst();
                    particleVelocity.addAssign(toNearest.normalize().mul(containmentStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply color-based sphere containment when in color spheres mode (skip for hidden particles)
            If(this.uniforms.colorSpheresMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                
                // Find closest palette color
                let minDist = float(999.0).toVar();
                let closestColorIndex = int(0).toVar();
                
                // Only check unique colors (10 unique colors in the palette)
                Loop({ start: 0, end: 10, type: 'int', name: 'i', condition: '<' }, ({i}) => {
                    // Access palette color from array using select
                    const paletteColor = select(
                        i.equal(0), this.uniforms.paletteColors[0],
                        select(i.equal(1), this.uniforms.paletteColors[1],
                        select(i.equal(2), this.uniforms.paletteColors[2],
                        select(i.equal(3), this.uniforms.paletteColors[3],
                        select(i.equal(4), this.uniforms.paletteColors[4],
                        select(i.equal(5), this.uniforms.paletteColors[5],
                        select(i.equal(6), this.uniforms.paletteColors[6],
                        select(i.equal(7), this.uniforms.paletteColors[7],
                        select(i.equal(8), this.uniforms.paletteColors[8],
                        this.uniforms.paletteColors[9]))))))))).toConst();
                    const colorDiff = particleColor.sub(paletteColor).toConst();
                    const dist = colorDiff.length().toConst();
                    If(dist.lessThan(minDist), () => {
                        minDist.assign(dist);
                        closestColorIndex.assign(i);
                    });
                });
                
                // Get sphere position for this color using same select pattern (only 10 unique colors)
                const targetSphere = select(
                    closestColorIndex.equal(0), this.uniforms.colorSpherePositions[0],
                    select(closestColorIndex.equal(1), this.uniforms.colorSpherePositions[1],
                    select(closestColorIndex.equal(2), this.uniforms.colorSpherePositions[2],
                    select(closestColorIndex.equal(3), this.uniforms.colorSpherePositions[3],
                    select(closestColorIndex.equal(4), this.uniforms.colorSpherePositions[4],
                    select(closestColorIndex.equal(5), this.uniforms.colorSpherePositions[5],
                    select(closestColorIndex.equal(6), this.uniforms.colorSpherePositions[6],
                    select(closestColorIndex.equal(7), this.uniforms.colorSpherePositions[7],
                    select(closestColorIndex.equal(8), this.uniforms.colorSpherePositions[8],
                    this.uniforms.colorSpherePositions[9]))))))))).toConst();
                const toSphere = particlePosition.sub(targetSphere).toConst();
                const distToSphere = toSphere.length().toConst();
                
                // If particle is outside sphere radius, push it back in
                If(distToSphere.greaterThan(this.uniforms.colorSphereRadius), () => {
                    const towardSphere = toSphere.normalize().negate().toConst();
                    const excessDistance = distToSphere.sub(this.uniforms.colorSphereRadius).toConst();
                    // Apply eased boost strength during boost period
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.colorSpheresBoostTime.toConst();
                    // Normalize boost time to 0-1 range
                    const boostProgress = boostTime.div(this.uniforms.colorSpheresBoostDuration).clamp(0.0, 1.0).toConst();
                    // Use smoothstep for ease-in-out easing
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst(); // smoothstep
                    // Interpolate between boost multiplier (200.0) and normal (1.0)
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardSphere.mul(containmentStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment pink mode: sphere containment for all particles, separate smaller sphere for pink particles (skip for hidden particles)
            If(this.uniforms.populationSegmentPinkMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                
                // Pink color: #FF56B7 = RGB(255/255, 86/255, 183/255) = RGB(1.0, 0.3372549, 0.717647)
                const pinkColor = vec3(1.0, 0.3372549, 0.717647).toConst();
                const colorDiff = particleColor.sub(pinkColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isPink = colorDist.lessThan(0.1).toConst(); // Threshold for pink detection
                
                // Main sphere containment for all particles with boost (same as color spheres)
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                
                // Use smaller radius for main sphere in population segment pink mode
                const mainSphereRadius = this.uniforms.populationSegmentPinkMainSphereRadius.toConst();
                
                // If particle is outside main sphere radius, push it back in
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    // Apply eased boost strength during boost period (same as color spheres)
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentPinkBoostTime.toConst();
                    // Normalize boost time to 0-1 range
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentPinkBoostDuration).clamp(0.0, 1.0).toConst();
                    // Use smoothstep for ease-in-out easing
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst(); // smoothstep
                    // Interpolate between boost multiplier (20.0) and normal (1.0)
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                // Additional containment for pink particles in smaller sphere
                If(isPink, () => {
                    const pinkSphereCenter = this.uniforms.pinkSphereCenter.toConst();
                    const toPinkSphere = particlePosition.sub(pinkSphereCenter).toConst();
                    const distToPinkSphere = toPinkSphere.length().toConst();
                    
                    // If pink particle is outside pink sphere radius, push it back in
                    If(distToPinkSphere.greaterThan(this.uniforms.pinkSphereRadius), () => {
                        const towardPinkSphere = toPinkSphere.normalize().negate().toConst();
                        const excessDistance = distToPinkSphere.sub(this.uniforms.pinkSphereRadius).toConst();
                        // Apply eased boost strength during boost period (same as color spheres)
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentPinkBoostTime.toConst();
                        // Normalize boost time to 0-1 range
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentPinkBoostDuration).clamp(0.0, 1.0).toConst();
                        // Use smoothstep for ease-in-out easing
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst(); // smoothstep
                        // Interpolate between boost multiplier (20.0) and normal (1.0)
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardPinkSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                // Negative space: push particles away from the negative space sphere (creates a chunk taken out)
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenter.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                
                // If particle is inside negative space sphere, push it away
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst(); // Direction away from center
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst(); // How deep inside
                    // Apply eased boost strength during boost period (same as color spheres)
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentPinkBoostTime.toConst();
                    // Normalize boost time to 0-1 range
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentPinkBoostDuration).clamp(0.0, 1.0).toConst();
                    // Use smoothstep for ease-in-out easing
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst(); // smoothstep
                    // Interpolate between boost multiplier (20.0) and normal (1.0)
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment purple mode (skip for hidden particles)
            If(this.uniforms.populationSegmentPurpleMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Purple color: #BC96F9 = RGB(188/255, 150/255, 249/255) = RGB(0.737, 0.588, 0.976)
                const purpleColor = vec3(0.737, 0.588, 0.976).toConst();
                const colorDiff = particleColor.sub(purpleColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isPurple = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentPurpleMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentPurpleBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isPurple, () => {
                    const purpleSphereCenter = this.uniforms.purpleSphereCenter.toConst();
                    const toPurpleSphere = particlePosition.sub(purpleSphereCenter).toConst();
                    const distToPurpleSphere = toPurpleSphere.length().toConst();
                    If(distToPurpleSphere.greaterThan(this.uniforms.purpleSphereRadius), () => {
                        const towardPurpleSphere = toPurpleSphere.normalize().negate().toConst();
                        const excessDistance = distToPurpleSphere.sub(this.uniforms.purpleSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentPurpleBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardPurpleSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterPurple.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentPurpleBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment cyan mode (skip for hidden particles)
            If(this.uniforms.populationSegmentCyanMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Cyan color: #C5F0F2 = RGB(197/255, 240/255, 242/255) = RGB(0.773, 0.941, 0.949)
                const cyanColor = vec3(0.773, 0.941, 0.949).toConst();
                const colorDiff = particleColor.sub(cyanColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isCyan = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentCyanMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentCyanBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentCyanBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isCyan, () => {
                    const cyanSphereCenter = this.uniforms.cyanSphereCenter.toConst();
                    const toCyanSphere = particlePosition.sub(cyanSphereCenter).toConst();
                    const distToCyanSphere = toCyanSphere.length().toConst();
                    If(distToCyanSphere.greaterThan(this.uniforms.cyanSphereRadius), () => {
                        const towardCyanSphere = toCyanSphere.normalize().negate().toConst();
                        const excessDistance = distToCyanSphere.sub(this.uniforms.cyanSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentCyanBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentCyanBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardCyanSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterCyan.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentCyanBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentCyanBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment teal mode (skip for hidden particles)
            If(this.uniforms.populationSegmentTealMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Teal color: #007487 = RGB(0/255, 116/255, 135/255) = RGB(0.0, 0.455, 0.529)
                const tealColor = vec3(0.0, 0.455, 0.529).toConst();
                const colorDiff = particleColor.sub(tealColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isTeal = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentTealMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentTealBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentTealBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isTeal, () => {
                    const tealSphereCenter = this.uniforms.tealSphereCenter.toConst();
                    const toTealSphere = particlePosition.sub(tealSphereCenter).toConst();
                    const distToTealSphere = toTealSphere.length().toConst();
                    If(distToTealSphere.greaterThan(this.uniforms.tealSphereRadius), () => {
                        const towardTealSphere = toTealSphere.normalize().negate().toConst();
                        const excessDistance = distToTealSphere.sub(this.uniforms.tealSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentTealBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentTealBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardTealSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterTeal.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentTealBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentTealBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment dark blue mode (skip for hidden particles)
            If(this.uniforms.populationSegmentDarkBlueMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Dark blue color: #153943 = RGB(21/255, 57/255, 67/255) = RGB(0.082, 0.224, 0.263)
                const darkBlueColor = vec3(0.082, 0.224, 0.263).toConst();
                const colorDiff = particleColor.sub(darkBlueColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isDarkBlue = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentDarkBlueMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentDarkBlueBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkBlueBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isDarkBlue, () => {
                    const darkBlueSphereCenter = this.uniforms.darkBlueSphereCenter.toConst();
                    const toDarkBlueSphere = particlePosition.sub(darkBlueSphereCenter).toConst();
                    const distToDarkBlueSphere = toDarkBlueSphere.length().toConst();
                    If(distToDarkBlueSphere.greaterThan(this.uniforms.darkBlueSphereRadius), () => {
                        const towardDarkBlueSphere = toDarkBlueSphere.normalize().negate().toConst();
                        const excessDistance = distToDarkBlueSphere.sub(this.uniforms.darkBlueSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentDarkBlueBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkBlueBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardDarkBlueSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterDarkBlue.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentDarkBlueBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkBlueBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment dark purple mode (skip for hidden particles)
            If(this.uniforms.populationSegmentDarkPurpleMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Dark purple color: #232356 = RGB(35/255, 35/255, 86/255) = RGB(0.137, 0.137, 0.337)
                const darkPurpleColor = vec3(0.137, 0.137, 0.337).toConst();
                const colorDiff = particleColor.sub(darkPurpleColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isDarkPurple = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentDarkPurpleMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentDarkPurpleBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isDarkPurple, () => {
                    const darkPurpleSphereCenter = this.uniforms.darkPurpleSphereCenter.toConst();
                    const toDarkPurpleSphere = particlePosition.sub(darkPurpleSphereCenter).toConst();
                    const distToDarkPurpleSphere = toDarkPurpleSphere.length().toConst();
                    If(distToDarkPurpleSphere.greaterThan(this.uniforms.darkPurpleSphereRadius), () => {
                        const towardDarkPurpleSphere = toDarkPurpleSphere.normalize().negate().toConst();
                        const excessDistance = distToDarkPurpleSphere.sub(this.uniforms.darkPurpleSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentDarkPurpleBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardDarkPurpleSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterDarkPurple.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentDarkPurpleBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment dark red mode (skip for hidden particles)
            If(this.uniforms.populationSegmentDarkRedMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Dark red color: #50121A = RGB(80/255, 18/255, 26/255) = RGB(0.314, 0.071, 0.102)
                const darkRedColor = vec3(0.314, 0.071, 0.102).toConst();
                const colorDiff = particleColor.sub(darkRedColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isDarkRed = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentDarkRedMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentDarkRedBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkRedBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isDarkRed, () => {
                    const darkRedSphereCenter = this.uniforms.darkRedSphereCenter.toConst();
                    const toDarkRedSphere = particlePosition.sub(darkRedSphereCenter).toConst();
                    const distToDarkRedSphere = toDarkRedSphere.length().toConst();
                    If(distToDarkRedSphere.greaterThan(this.uniforms.darkRedSphereRadius), () => {
                        const towardDarkRedSphere = toDarkRedSphere.normalize().negate().toConst();
                        const excessDistance = distToDarkRedSphere.sub(this.uniforms.darkRedSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentDarkRedBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkRedBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardDarkRedSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterDarkRed.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentDarkRedBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkRedBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment orange mode (skip for hidden particles)
            If(this.uniforms.populationSegmentOrangeMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Orange color: #FF6425 = RGB(255/255, 100/255, 37/255) = RGB(1.0, 0.392, 0.145)
                const orangeColor = vec3(1.0, 0.392, 0.145).toConst();
                const colorDiff = particleColor.sub(orangeColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isOrange = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentOrangeMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentOrangeBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentOrangeBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isOrange, () => {
                    const orangeSphereCenter = this.uniforms.orangeSphereCenter.toConst();
                    const toOrangeSphere = particlePosition.sub(orangeSphereCenter).toConst();
                    const distToOrangeSphere = toOrangeSphere.length().toConst();
                    If(distToOrangeSphere.greaterThan(this.uniforms.orangeSphereRadius), () => {
                        const towardOrangeSphere = toOrangeSphere.normalize().negate().toConst();
                        const excessDistance = distToOrangeSphere.sub(this.uniforms.orangeSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentOrangeBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentOrangeBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardOrangeSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterOrange.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentOrangeBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentOrangeBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            
            // Apply population segment light green mode (skip for hidden particles)
            If(this.uniforms.populationSegmentLightGreenMode.equal(uint(1)).and(shouldHide.not()), () => {
                const particleColor = this.particleBuffer.element(instanceIndex).get('color').xyz.toConst();
                // Light green color: #D6F499 = RGB(214/255, 244/255, 153/255) = RGB(0.839, 0.957, 0.600)
                const lightGreenColor = vec3(0.839, 0.957, 0.600).toConst();
                const colorDiff = particleColor.sub(lightGreenColor).toConst();
                const colorDist = colorDiff.length().toConst();
                const isLightGreen = colorDist.lessThan(0.1).toConst();
                
                const sphereCenter = vec3(this.uniforms.sphereCenterX, this.uniforms.sphereCenterY, this.uniforms.sphereCenterZ).toConst();
                const toCenter = particlePosition.sub(sphereCenter).toConst();
                const distToCenter = toCenter.length().toConst();
                const mainSphereRadius = this.uniforms.populationSegmentLightGreenMainSphereRadius.toConst();
                
                If(distToCenter.greaterThan(mainSphereRadius), () => {
                    const towardCenter = toCenter.normalize().negate().toConst();
                    const excessDistance = distToCenter.sub(mainSphereRadius).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentLightGreenBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentLightGreenBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(towardCenter.mul(containmentStrength).mul(this.uniforms.dt));
                });
                
                If(isLightGreen, () => {
                    const lightGreenSphereCenter = this.uniforms.lightGreenSphereCenter.toConst();
                    const toLightGreenSphere = particlePosition.sub(lightGreenSphereCenter).toConst();
                    const distToLightGreenSphere = toLightGreenSphere.length().toConst();
                    If(distToLightGreenSphere.greaterThan(this.uniforms.lightGreenSphereRadius), () => {
                        const towardLightGreenSphere = toLightGreenSphere.normalize().negate().toConst();
                        const excessDistance = distToLightGreenSphere.sub(this.uniforms.lightGreenSphereRadius).toConst();
                        const baseStrength = float(0.3).toConst();
                        const boostTime = this.uniforms.populationSegmentLightGreenBoostTime.toConst();
                        const boostProgress = boostTime.div(this.uniforms.populationSegmentLightGreenBoostDuration).clamp(0.0, 1.0).toConst();
                        const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                        const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                        const containmentStrength = excessDistance.mul(baseStrength).mul(boostMultiplier).toConst();
                        particleVelocity.addAssign(towardLightGreenSphere.mul(containmentStrength).mul(this.uniforms.dt));
                    });
                });
                
                const negativeSpaceCenter = this.uniforms.negativeSpaceSphereCenterLightGreen.toConst();
                const toNegativeSpace = particlePosition.sub(negativeSpaceCenter).toConst();
                const distToNegativeSpace = toNegativeSpace.length().toConst();
                If(distToNegativeSpace.lessThan(this.uniforms.negativeSpaceSphereRadius), () => {
                    const awayFromNegativeSpace = toNegativeSpace.normalize().toConst();
                    const depthInside = this.uniforms.negativeSpaceSphereRadius.sub(distToNegativeSpace).toConst();
                    const baseStrength = float(0.3).toConst();
                    const boostTime = this.uniforms.populationSegmentLightGreenBoostTime.toConst();
                    const boostProgress = boostTime.div(this.uniforms.populationSegmentLightGreenBoostDuration).clamp(0.0, 1.0).toConst();
                    const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                    const boostMultiplier = mix(float(1.0), float(20.0), easedProgress).toConst();
                    const repulsionStrength = depthInside.mul(baseStrength).mul(boostMultiplier).toConst();
                    particleVelocity.addAssign(awayFromNegativeSpace.mul(repulsionStrength).mul(this.uniforms.dt));
                });
            });
            

            // Only apply noise when not in grid mode and not in front gravity mode (sphere containment and color spheres modes keep noise)
            If(this.uniforms.gridMode.equal(uint(0)).and(this.uniforms.frontGravityMode.equal(uint(0))), () => {
                const noise = triNoise3Dvec(particlePosition.mul(0.015), time, 0.11).sub(0.285).normalize().mul(0.28).toVar();
                particleVelocity.subAssign(noise.mul(this.uniforms.noise).mul(this.uniforms.dt));
            });

            const cellIndex =  ivec3(particlePosition).sub(1).toConst("cellIndex");
            const cellDiff = particlePosition.fract().sub(0.5).toConst("cellDiff");

            const w0 = float(0.5).mul(float(0.5).sub(cellDiff)).mul(float(0.5).sub(cellDiff));
            const w1 = float(0.75).sub(cellDiff.mul(cellDiff));
            const w2 = float(0.5).mul(float(0.5).add(cellDiff)).mul(float(0.5).add(cellDiff));
            const weights = array([w0,w1,w2]).toConst("weights");

            const B = mat3(0).toVar("B");
            // Only apply fluid simulation forces when not in grid mode (front gravity mode keeps them) (skip for hidden particles)
            If(this.uniforms.gridMode.equal(uint(0)).and(shouldHide.not()), () => {
                Loop({ start: 0, end: 3, type: 'int', name: 'gx', condition: '<' }, ({gx}) => {
                    Loop({ start: 0, end: 3, type: 'int', name: 'gy', condition: '<' }, ({gy}) => {
                        Loop({ start: 0, end: 3, type: 'int', name: 'gz', condition: '<' }, ({gz}) => {
                            const weight = weights.element(gx).x.mul(weights.element(gy).y).mul(weights.element(gz).z);
                            const cellX = cellIndex.add(ivec3(gx,gy,gz)).toConst();
                            const cellDist = vec3(cellX).add(0.5).sub(particlePosition).toConst("cellDist");
                            const cellPtr = getCellPtr(cellX);

                            const weightedVelocity = this.cellBufferF.element(cellPtr).xyz.mul(weight).toConst("weightedVelocity");
                            const term = mat3(
                                weightedVelocity.mul(cellDist.x),
                                weightedVelocity.mul(cellDist.y),
                                weightedVelocity.mul(cellDist.z)
                            );
                            B.addAssign(term);
                            particleVelocity.addAssign(weightedVelocity);
                        });
                    });
                });
            });

            // Only apply mouse force when not in grid mode (front gravity mode keeps it) and cursor interaction is enabled (skip for hidden particles)
            If(this.uniforms.gridMode.equal(uint(0)).and(this.uniforms.cursorInteraction.equal(uint(1))).and(shouldHide.not()), () => {
                const dist = cross(this.uniforms.mouseRayDirection, particlePosition.mul(vec3(1,1,0.4)).sub(this.uniforms.mouseRayOrigin)).length()
                const force = dist.mul(0.1).oneMinus().max(0.0).pow(2);
                particleVelocity.addAssign(this.uniforms.mouseForce.mul(1).mul(force));
                particleVelocity.mulAssign(particleMass); // to ensure difference between particles
            });

            this.particleBuffer.element(instanceIndex).get('C').assign(B.mul(4));
            
            // Apply damping in front gravity mode (or cylinder mode) or color spheres mode to reduce inertia
            If(this.uniforms.frontGravityMode.equal(uint(1)), () => {
                const dampingFactor = float(0.98).toConst(); // Reduce velocity by 2% each frame
                particleVelocity.mulAssign(dampingFactor);
            });
            // Color spheres mode with boost period damping (skip for hidden particles)
            If(this.uniforms.colorSpheresMode.equal(uint(1)).and(shouldHide.not()), () => {
                // Ease damping in and out during boost period
                const boostTime = this.uniforms.colorSpheresBoostTime.toConst();
                // Normalize boost time to 0-1 range
                const boostProgress = boostTime.div(this.uniforms.colorSpheresBoostDuration).clamp(0.0, 1.0).toConst();
                // Use smoothstep for ease-in-out easing
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst(); // smoothstep
                // Interpolate between boost damping (0.7) and normal damping (0.98)
                const boostDamping = float(0.7).toConst(); // Stronger damping during boost
                const normalDamping = float(0.98).toConst(); // Normal damping
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment pink mode with boost period damping (same as color spheres) (skip for hidden particles)
            If(this.uniforms.populationSegmentPinkMode.equal(uint(1)).and(shouldHide.not()), () => {
                // Ease damping in and out during boost period
                const boostTime = this.uniforms.populationSegmentPinkBoostTime.toConst();
                // Normalize boost time to 0-1 range
                const boostProgress = boostTime.div(this.uniforms.populationSegmentPinkBoostDuration).clamp(0.0, 1.0).toConst();
                // Use smoothstep for ease-in-out easing
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst(); // smoothstep
                // Interpolate between boost damping (0.7) and normal damping (0.98)
                const boostDamping = float(0.7).toConst(); // Stronger damping during boost
                const normalDamping = float(0.98).toConst(); // Normal damping
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment purple mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentPurpleMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentPurpleBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment cyan mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentCyanMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentCyanBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentCyanBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment teal mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentTealMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentTealBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentTealBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment dark blue mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentDarkBlueMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentDarkBlueBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkBlueBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment dark purple mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentDarkPurpleMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentDarkPurpleBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkPurpleBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment dark red mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentDarkRedMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentDarkRedBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentDarkRedBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment orange mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentOrangeMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentOrangeBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentOrangeBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            // Population segment light green mode with boost period damping (skip for hidden particles)
            If(this.uniforms.populationSegmentLightGreenMode.equal(uint(1)).and(shouldHide.not()), () => {
                const boostTime = this.uniforms.populationSegmentLightGreenBoostTime.toConst();
                const boostProgress = boostTime.div(this.uniforms.populationSegmentLightGreenBoostDuration).clamp(0.0, 1.0).toConst();
                const easedProgress = boostProgress.mul(boostProgress).mul(float(3.0).sub(boostProgress.mul(2.0))).toConst();
                const boostDamping = float(0.7).toConst();
                const normalDamping = float(0.98).toConst();
                const dampingFactor = mix(normalDamping, boostDamping, easedProgress).toConst();
                particleVelocity.mulAssign(dampingFactor);
            });
            
            // Hide particles in vertical slices at x edges based on percentage
            If(shouldHide, () => {
                // Calculate the width of the edge regions (10% of box width)
                const edgeRegionWidth = this.uniforms.gridSize.x.mul(0.1).toConst();
                const leftEdgeMax = edgeRegionWidth.toConst();
                const rightEdgeMin = this.uniforms.gridSize.x.sub(edgeRegionWidth).toConst();
                
                // Use hash to deterministically assign particles to left or right edge
                const hash = float(instanceIndex).mul(12.9898).toConst();
                const useLeftEdge = hash.fract().lessThan(0.5).toConst();
                
                // Target x position: center of the edge region
                const targetX = select(
                    useLeftEdge,
                    leftEdgeMax.div(2.0), // Center of left edge region
                    rightEdgeMin.add(edgeRegionWidth.div(2.0)) // Center of right edge region
                ).toConst();
                
                // Calculate distance from target x position
                const distFromTargetX = particlePosition.x.sub(targetX).abs().toConst();
                
                // If particle is outside the edge region, push it in
                If(distFromTargetX.greaterThan(edgeRegionWidth.div(2.0)), () => {
                    // Calculate direction toward target x
                    const towardTargetX = select(
                        particlePosition.x.lessThan(targetX),
                        float(1.0), // Move right
                        float(-1.0) // Move left
                    ).toConst();
                    
                    // Strength increases with distance outside region
                    const excessDistance = distFromTargetX.sub(edgeRegionWidth.div(2.0)).toConst();
                    const containmentStrength = excessDistance.mul(0.3).toConst();
                    particleVelocity.x.addAssign(towardTargetX.mul(containmentStrength).mul(this.uniforms.dt));
                });
            });
            
            particlePosition.addAssign(particleVelocity.mul(this.uniforms.dt));
            particlePosition.assign(clamp(particlePosition, vec3(2), this.uniforms.gridSize.sub(2)));

            const wallStiffness = 0.3;
            const xN = particlePosition.add(particleVelocity.mul(this.uniforms.dt).mul(3.0)).toConst("xN");
            const wallMin = vec3(3).toConst("wallMin");
            const wallMax = vec3(this.uniforms.gridSize).sub(3).toConst("wallMax");
            If(xN.x.lessThan(wallMin.x), () => { particleVelocity.x.addAssign(wallMin.x.sub(xN.x).mul(wallStiffness)); });
            If(xN.x.greaterThan(wallMax.x), () => { particleVelocity.x.addAssign(wallMax.x.sub(xN.x).mul(wallStiffness)); });
            If(xN.y.lessThan(wallMin.y), () => { particleVelocity.y.addAssign(wallMin.y.sub(xN.y).mul(wallStiffness)); });
            If(xN.y.greaterThan(wallMax.y), () => { particleVelocity.y.addAssign(wallMax.y.sub(xN.y).mul(wallStiffness)); });
            If(xN.z.lessThan(wallMin.z), () => { particleVelocity.z.addAssign(wallMin.z.sub(xN.z).mul(wallStiffness)); });
            If(xN.z.greaterThan(wallMax.z), () => { particleVelocity.z.addAssign(wallMax.z.sub(xN.z).mul(wallStiffness)); });

            this.particleBuffer.element(instanceIndex).get('position').assign(particlePosition)
            this.particleBuffer.element(instanceIndex).get('velocity').assign(particleVelocity)

            const direction = this.particleBuffer.element(instanceIndex).get('direction');
            direction.assign(mix(direction,particleVelocity, 0.1));

            // Color is set once during initialization and should not be modified here
        })().compute(1);
    }

    setMouseRay(origin, direction, pos) {
        origin.multiplyScalar(64);
        pos.multiplyScalar(64);
        origin.add(new THREE.Vector3(64,0,0)); // Updated for 2x wider container
        this.uniforms.mouseRayDirection.value.copy(direction.normalize());
        this.uniforms.mouseRayOrigin.value.copy(origin);
        this.mousePos.copy(pos);
    }

    async loadImageAndSetupGrid(imagePath) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                console.log("Image loaded:", img.width, "x", img.height);
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                this.imageData = ctx.getImageData(0, 0, img.width, img.height);
                console.log("Image data extracted, first pixel:", 
                    this.imageData.data[0], this.imageData.data[1], this.imageData.data[2]);
                
                // Calculate grid dimensions based on number of particles
                // Ensure gridWidth * gridHeight >= numParticles to avoid position collisions
                const numParticles = conf.particles;
                const aspectRatio = img.width / img.height;
                
                // Start with ideal dimensions based on aspect ratio
                this.gridHeight = Math.floor(Math.sqrt(numParticles / aspectRatio));
                this.gridWidth = Math.floor(this.gridHeight * aspectRatio);
                
                // Ensure we have enough grid positions for all particles
                // Increase dimensions while maintaining aspect ratio until we have enough
                while (this.gridWidth * this.gridHeight < numParticles) {
                    // Increase the dimension that needs it most based on aspect ratio
                    if (this.gridWidth / this.gridHeight < aspectRatio) {
                        this.gridWidth++;
                    } else {
                        this.gridHeight++;
                    }
                }
                
                console.log("Grid dimensions:", this.gridWidth, "x", this.gridHeight, "for", numParticles, "particles");
                console.log("Grid capacity:", this.gridWidth * this.gridHeight, "positions");
                
                // Store target positions and colors for each particle
                this.gridTargetPositions = [];
                this.gridTargetColors = [];
                
                // Calculate spacing in grid space - use equal spacing for both x and y
                // Use the smaller dimension to ensure square spacing
                const availableWidth = 120; // Use most of the 128 width
                const availableHeight = 60; // Use most of the 64 height
                const spacingX = availableWidth / this.gridWidth;
                const spacingY = availableHeight / this.gridHeight;
                // Use the smaller spacing to ensure equal spacing in both dimensions
                const equalSpacing = Math.min(spacingX, spacingY);
                this.gridSpacingX = equalSpacing;
                this.gridSpacingY = equalSpacing;
                // Center the grid in both dimensions
                this.gridStartX = (128 - (this.gridWidth * this.gridSpacingX)) / 2;
                this.gridStartY = (64 - (this.gridHeight * this.gridSpacingY)) / 2;
                const zPos = 32; // Center in z dimension
                
                for (let i = 0; i < numParticles; i++) {
                    const gridX = i % this.gridWidth;
                    const gridY = Math.floor(i / this.gridWidth);
                    
                    // Ensure we don't exceed grid bounds (shouldn't happen now, but safety check)
                    if (gridY >= this.gridHeight) {
                        console.warn(`Particle ${i} would exceed grid height ${this.gridHeight}`);
                        continue;
                    }
                    
                    // Calculate target position in grid space
                    const targetX = this.gridStartX + gridX * this.gridSpacingX;
                    const targetY = this.gridStartY + gridY * this.gridSpacingY;
                    this.gridTargetPositions.push(new THREE.Vector3(targetX, targetY, zPos));
                    
                    // Sample pixel color from image at grid resolution
                    // Scale image coordinates to match grid dimensions (not original image dimensions)
                    // Flip both X and Y coordinates to match the correct orientation
                    const normalizedX = this.gridWidth > 1 ? gridX / (this.gridWidth - 1) : 0;
                    const normalizedY = this.gridHeight > 1 ? gridY / (this.gridHeight - 1) : 0;
                    const imgX = Math.floor((1.0 - normalizedX) * (img.width - 1));
                    const imgY = Math.floor((1.0 - normalizedY) * (img.height - 1));
                    const pixelIndex = (imgY * img.width + imgX) * 4;
                    const r = this.imageData.data[pixelIndex] / 255;
                    const g = this.imageData.data[pixelIndex + 1] / 255;
                    const b = this.imageData.data[pixelIndex + 2] / 255;
                    this.gridTargetColors.push(new THREE.Vector3(r, g, b));
                }
                
                console.log("Extracted", this.gridTargetColors.length, "colors. First few:", 
                    this.gridTargetColors.slice(0, 3).map(c => `rgb(${Math.round(c.x*255)}, ${Math.round(c.y*255)}, ${Math.round(c.z*255)})`));
                
                // Update uniforms
                this.uniforms.gridWidth.value = this.gridWidth;
                this.uniforms.gridHeight.value = this.gridHeight;
                this.uniforms.gridStartX.value = this.gridStartX;
                this.uniforms.gridStartY.value = this.gridStartY;
                this.uniforms.gridSpacingX.value = this.gridSpacingX;
                this.uniforms.gridSpacingY.value = this.gridSpacingY;
                
                resolve();
            };
            img.onerror = reject;
            img.src = imagePath;
        });
    }

    setupPolygonContainment() {
        // Parse SVG path to extract vertices
        // SVG viewBox: 0 0 559 302
        // Path: M311.096 125.894L92.6019 0C62.0821 98.4958 0.834339 296.632 0.00197867 301.21C-0.830382 305.788 261.155 235.488 392.251 199.766L520.226 283.002L558.723 4.16184L311.096 125.894Z
        
        // Extract key vertices from the path
        // For curves, we'll use the curve endpoints
        const svgViewBox = { width: 559, height: 302 };
        const svgVertices = [
            { x: 311.096, y: 125.894 },  // Start
            { x: 92.6019, y: 0 },        // Line
            { x: 0.834339, y: 296.632 }, // Curve endpoint
            { x: 261.155, y: 235.488 },  // Curve endpoint
            { x: 392.251, y: 199.766 },  // Line
            { x: 520.226, y: 283.002 },  // Line
            { x: 558.723, y: 4.16184 }, // Line
            // Close back to start
        ];
        
        // Normalize to simulation space (128x64)
        // Map from SVG coordinates (0-559, 0-302) to simulation (0-128, 0-64)
        // Center and scale to fit nicely in the simulation space
        const simWidth = 120; // Use most of 128 width
        const simHeight = 60; // Use most of 64 height
        const simStartX = (128 - simWidth) / 2;
        const simStartY = (64 - simHeight) / 2;
        
        this.polygonVertices = [];
        for (const vertex of svgVertices) {
            // Normalize to 0-1 range
            const normalizedX = vertex.x / svgViewBox.width;
            const normalizedY = vertex.y / svgViewBox.height;
            // Map to simulation space
            const simX = simStartX + normalizedX * simWidth;
            const simY = simStartY + normalizedY * simHeight;
            this.polygonVertices.push(new THREE.Vector2(simX, simY));
        }
        
        // Update uniforms
        this.uniforms.polygonVertexCount.value = this.polygonVertices.length;
        for (let i = 0; i < 16; i++) {
            if (i < this.polygonVertices.length) {
                this.uniforms.polygonVertices[i].value.copy(this.polygonVertices[i]);
            } else {
                // Fill remaining with zero
                this.uniforms.polygonVertices[i].value.set(0, 0);
            }
        }
        
        console.log(`Polygon containment: set up ${this.polygonVertices.length} vertices`);
    }

    setupColorSpheres() {
        // Get full palette colors (with duplicates for particle assignment)
        const palette = [
            "#FF56B7", "#BC96F9", "#C5F0F2", "#007487", "#007487",
            "#153943", "#232356", "#50121A", "#FF6425", "#D6F499",
            "#5CCAD8", "#5CCAD8", "#C5F0F2",
        ];
        this.paletteRGB = palette.map(hex => {
            const r = parseInt(hex.slice(1, 3), 16) / 255;
            const g = parseInt(hex.slice(3, 5), 16) / 255;
            const b = parseInt(hex.slice(5, 7), 16) / 255;
            return new THREE.Vector3(r, g, b);
        });
        
        // Get unique colors for sphere positions only
        const uniquePalette = [];
        const seenColors = new Set();
        const uniquePaletteRGB = [];
        
        for (let i = 0; i < palette.length; i++) {
            if (!seenColors.has(palette[i])) {
                seenColors.add(palette[i]);
                uniquePalette.push(palette[i]);
                uniquePaletteRGB.push(this.paletteRGB[i]);
            }
        }
        
        // Arrange spheres in 3 rows: 3, 4, 3 spheres (for unique colors only)
        const rowLayouts = [3, 4, 3]; // Spheres per row
        const spacingX = 21;
        const spacingY = 21;
        const boxCenterX = 64; // Center of box width
        const boxCenterY = 32; // Center of box height
        const boxCenterZ = 32; // Center of box depth
        const totalRows = rowLayouts.length;
        
        // Calculate total grid height to center vertically
        const totalGridHeight = (totalRows - 1) * spacingY;
        const startY = boxCenterY - totalGridHeight / 2;
        
        this.colorSpherePositions = [];
        let colorIndex = 0;
        
        for (let row = 0; row < rowLayouts.length; row++) {
            const spheresInRow = rowLayouts[row];
            // Calculate row width to center horizontally
            const rowWidth = (spheresInRow - 1) * spacingX;
            const startX = boxCenterX - rowWidth / 2;
            const y = startY + row * spacingY;
            
            for (let col = 0; col < spheresInRow && colorIndex < uniquePaletteRGB.length; col++) {
                const x = startX + col * spacingX;
                this.colorSpherePositions.push(new THREE.Vector3(x, y, boxCenterZ));
                colorIndex++;
            }
        }
        
        // Store unique colors for sphere matching (particles will match to these)
        this.uniquePaletteRGB = uniquePaletteRGB;
        
        // Update uniforms - use unique colors for sphere matching
        // We need to fill all 13 uniform slots, but only use unique colors for matching
        for (let i = 0; i < 13; i++) {
            if (i < this.colorSpherePositions.length) {
                this.uniforms.colorSpherePositions[i].value.copy(this.colorSpherePositions[i]);
            }
            // For palette colors, use unique colors for matching (repeat last if needed)
            const uniqueIndex = Math.min(i, this.uniquePaletteRGB.length - 1);
            this.uniforms.paletteColors[i].value.copy(this.uniquePaletteRGB[uniqueIndex]);
        }
    }

    async toggleGridMode(mode) {
        // mode: 0 = chaos, 1 = canvasser, 2 = protest, 3 = front gravity, 4 = front gravity + cylinder, 5 = sphere containment, 6 = color spheres, 7 = usa, 8 = polygon containment, 9 = population segment pink, 10 = purple, 11 = cyan, 12 = teal
        const enabled = (mode > 0 && mode < 3) || mode === 7; // Modes 1, 2, and 7 are grid modes
        const frontGravityEnabled = mode === 3 || mode === 4;
        const cylinderEnabled = mode === 4;
        const sphereContainmentEnabled = mode === 5;
        const polygonContainmentEnabled = mode === 8;
        const colorSpheresEnabled = mode === 6;
        const populationSegmentPinkEnabled = mode === 9;
        const populationSegmentPurpleEnabled = mode === 10;
        const populationSegmentCyanEnabled = mode === 11;
        const populationSegmentTealEnabled = mode === 12;
        const populationSegmentDarkBlueEnabled = mode === 13;
        const populationSegmentDarkPurpleEnabled = mode === 14;
        const populationSegmentDarkRedEnabled = mode === 15;
        const populationSegmentOrangeEnabled = mode === 16;
        const populationSegmentLightGreenEnabled = mode === 17;
        const twoColorSphereEnabled = mode === 18;
        this.gridMode = enabled;
        this.uniforms.gridMode.value = enabled ? 1 : 0;
        this.uniforms.frontGravityMode.value = frontGravityEnabled ? 1 : 0;
        this.uniforms.cylinderMode.value = cylinderEnabled ? 1 : 0;
        this.uniforms.sphereContainmentMode.value = sphereContainmentEnabled ? 1 : 0;
        this.uniforms.polygonContainmentMode.value = polygonContainmentEnabled ? 1 : 0;
        this.uniforms.colorSpheresMode.value = colorSpheresEnabled ? 1 : 0;
        
        console.log('toggleGridMode - mode:', mode, 'frontGravityEnabled:', frontGravityEnabled, 'uniform value:', this.uniforms.frontGravityMode.value);
        this.uniforms.populationSegmentPinkMode.value = populationSegmentPinkEnabled ? 1 : 0;
        this.uniforms.populationSegmentPurpleMode.value = populationSegmentPurpleEnabled ? 1 : 0;
        this.uniforms.populationSegmentCyanMode.value = populationSegmentCyanEnabled ? 1 : 0;
        this.uniforms.populationSegmentTealMode.value = populationSegmentTealEnabled ? 1 : 0;
        this.uniforms.populationSegmentDarkBlueMode.value = populationSegmentDarkBlueEnabled ? 1 : 0;
        this.uniforms.populationSegmentDarkPurpleMode.value = populationSegmentDarkPurpleEnabled ? 1 : 0;
        this.uniforms.populationSegmentDarkRedMode.value = populationSegmentDarkRedEnabled ? 1 : 0;
        this.uniforms.populationSegmentOrangeMode.value = populationSegmentOrangeEnabled ? 1 : 0;
        this.uniforms.populationSegmentLightGreenMode.value = populationSegmentLightGreenEnabled ? 1 : 0;
        this.uniforms.twoColorSphereMode.value = twoColorSphereEnabled ? 1 : 0;
        this.uniforms.twoColorSphereMode.value = twoColorSphereEnabled ? 1 : 0;
        
        // Setup polygon containment if entering polygon mode
        if (polygonContainmentEnabled && this.polygonVertices.length === 0) {
            this.setupPolygonContainment();
        }
        
        // Setup color sphere positions if entering color spheres mode
        if (colorSpheresEnabled && this.colorSpherePositions.length === 0) {
            this.setupColorSpheres();
        }
        
        // Start boost timer when entering color spheres mode
        if (colorSpheresEnabled) {
            this.colorSpheresModeStartTime = Date.now();
            this.uniforms.colorSpheresBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.colorSpheresBoostTime.value = 0.0;
        }
        
        // Setup pink sphere position when entering population segment pink mode
        if (populationSegmentPinkEnabled) {
            // Calculate pink sphere center: 20 units away at 30 degrees in XY plane
            const angleRad = (30.0 * Math.PI) / 180.0; // 30 degrees in radians
            const offsetDistance = 40.0;
            const offsetX = offsetDistance * Math.cos(angleRad);
            const offsetY = offsetDistance * Math.sin(angleRad);
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            const pinkSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + offsetX,
                mainSphereCenter.y + offsetY,
                mainSphereCenter.z
            );
            this.uniforms.pinkSphereCenter.value.copy(pinkSphereCenter);
            
            // Calculate negative space sphere center: same angle, but 0.7x the distance
            const negativeSpaceDistance = offsetDistance * 0.5; // 0.7x as far away
            const negativeSpaceOffsetX = negativeSpaceDistance * Math.cos(angleRad);
            const negativeSpaceOffsetY = negativeSpaceDistance * Math.sin(angleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + negativeSpaceOffsetX,
                mainSphereCenter.y + negativeSpaceOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenter.value.copy(negativeSpaceSphereCenter);
            
            // Start boost timer
            this.populationSegmentPinkModeStartTime = Date.now();
            this.uniforms.populationSegmentPinkBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentPinkBoostTime.value = 0.0;
        }
        
        // Setup purple sphere position when entering population segment purple mode
        if (populationSegmentPurpleEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Purple sphere at 135 degrees, chunk at 110 degrees
            const sphereAngleRad = (135.0 * Math.PI) / 180.0;
            const chunkAngleRad = (110.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const purpleSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.purpleSphereCenter.value.copy(purpleSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterPurple.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentPurpleModeStartTime = Date.now();
            this.uniforms.populationSegmentPurpleBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentPurpleBoostTime.value = 0.0;
        }
        
        // Setup cyan sphere position when entering population segment cyan mode
        if (populationSegmentCyanEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Cyan sphere at 270 degrees, chunk at 250 degrees
            const sphereAngleRad = (270.0 * Math.PI) / 180.0;
            const chunkAngleRad = (250.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const cyanSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.cyanSphereCenter.value.copy(cyanSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterCyan.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentCyanModeStartTime = Date.now();
            this.uniforms.populationSegmentCyanBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentCyanBoostTime.value = 0.0;
        }
        
        // Setup teal sphere position when entering population segment teal mode
        if (populationSegmentTealEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Teal sphere at 45 degrees, chunk at 65 degrees
            const sphereAngleRad = (45.0 * Math.PI) / 180.0;
            const chunkAngleRad = (65.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const tealSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.tealSphereCenter.value.copy(tealSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterTeal.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentTealModeStartTime = Date.now();
            this.uniforms.populationSegmentTealBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentTealBoostTime.value = 0.0;
        }
        
        // Setup dark blue sphere position when entering population segment dark blue mode
        if (populationSegmentDarkBlueEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Dark blue sphere at 180 degrees, chunk at 200 degrees
            const sphereAngleRad = (180.0 * Math.PI) / 180.0;
            const chunkAngleRad = (200.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const darkBlueSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.darkBlueSphereCenter.value.copy(darkBlueSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterDarkBlue.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentDarkBlueModeStartTime = Date.now();
            this.uniforms.populationSegmentDarkBlueBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentDarkBlueBoostTime.value = 0.0;
        }
        
        // Setup dark purple sphere position when entering population segment dark purple mode
        if (populationSegmentDarkPurpleEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Dark purple sphere at 300 degrees, chunk at 320 degrees
            const sphereAngleRad = (300.0 * Math.PI) / 180.0;
            const chunkAngleRad = (320.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const darkPurpleSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.darkPurpleSphereCenter.value.copy(darkPurpleSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterDarkPurple.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentDarkPurpleModeStartTime = Date.now();
            this.uniforms.populationSegmentDarkPurpleBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentDarkPurpleBoostTime.value = 0.0;
        }
        
        // Setup dark red sphere position when entering population segment dark red mode
        if (populationSegmentDarkRedEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Dark red sphere at 0 degrees, chunk at 20 degrees
            const sphereAngleRad = (0.0 * Math.PI) / 180.0;
            const chunkAngleRad = (20.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const darkRedSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.darkRedSphereCenter.value.copy(darkRedSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterDarkRed.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentDarkRedModeStartTime = Date.now();
            this.uniforms.populationSegmentDarkRedBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentDarkRedBoostTime.value = 0.0;
        }
        
        // Setup orange sphere position when entering population segment orange mode
        if (populationSegmentOrangeEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Orange sphere at 90 degrees, chunk at 70 degrees
            const sphereAngleRad = (90.0 * Math.PI) / 180.0;
            const chunkAngleRad = (70.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const orangeSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.orangeSphereCenter.value.copy(orangeSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterOrange.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentOrangeModeStartTime = Date.now();
            this.uniforms.populationSegmentOrangeBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentOrangeBoostTime.value = 0.0;
        }
        
        // Setup light green sphere position when entering population segment light green mode
        if (populationSegmentLightGreenEnabled) {
            const mainSphereCenter = new THREE.Vector3(
                this.uniforms.sphereCenterX.value,
                this.uniforms.sphereCenterY.value,
                this.uniforms.sphereCenterZ.value
            );
            // Light green sphere at 210 degrees, chunk at 190 degrees
            const sphereAngleRad = (210.0 * Math.PI) / 180.0;
            const chunkAngleRad = (190.0 * Math.PI) / 180.0;
            const offsetDistance = 40.0;
            const sphereOffsetX = offsetDistance * Math.cos(sphereAngleRad);
            const sphereOffsetY = offsetDistance * Math.sin(sphereAngleRad);
            const lightGreenSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + sphereOffsetX,
                mainSphereCenter.y + sphereOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.lightGreenSphereCenter.value.copy(lightGreenSphereCenter);
            
            const negativeSpaceDistance = offsetDistance * 0.5;
            const chunkOffsetX = negativeSpaceDistance * Math.cos(chunkAngleRad);
            const chunkOffsetY = negativeSpaceDistance * Math.sin(chunkAngleRad);
            const negativeSpaceSphereCenter = new THREE.Vector3(
                mainSphereCenter.x + chunkOffsetX,
                mainSphereCenter.y + chunkOffsetY,
                mainSphereCenter.z
            );
            this.uniforms.negativeSpaceSphereCenterLightGreen.value.copy(negativeSpaceSphereCenter);
            
            this.populationSegmentLightGreenModeStartTime = Date.now();
            this.uniforms.populationSegmentLightGreenBoostTime.value = mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION;
        } else {
            this.uniforms.populationSegmentLightGreenBoostTime.value = 0.0;
        }
        
        // Get color member info once
        const colorMember = this.particleBuffer.layout["color"];
        const colorOffset = colorMember.offset;
        const numParticles = conf.particles;
        
        if (enabled) {
            // Determine which image to load based on mode
            let imageToLoad = null;
            if (mode === 1) {
                imageToLoad = canvasserImage;
            } else if (mode === 2) {
                imageToLoad = protestImage;
            } else if (mode === 7) {
                imageToLoad = usaImage;
            }
            
            // Load image and setup grid if not already loaded for this mode
            if (imageToLoad && this.currentImageMode !== mode) {
                await this.loadImageAndSetupGrid(imageToLoad);
                this.currentImageMode = mode;
            }
            
            // Update particle colors to image colors
            // Write directly to floatArray to ensure changes are reflected
            // Get palette colors for lerping
            const palette = [
                "#FF56B7", "#BC96F9", "#C5F0F2", "#007487", "#007487",
                "#153943", "#232356", "#50121A", "#FF6425", "#D6F499",
                "#5CCAD8", "#5CCAD8", "#C5F0F2",
            ];
            const paletteRGB = palette.map(hex => {
                const r = parseInt(hex.slice(1, 3), 16) / 255;
                const g = parseInt(hex.slice(3, 5), 16) / 255;
                const b = parseInt(hex.slice(5, 7), 16) / 255;
                return new THREE.Vector3(r, g, b);
            });
            
            // Helper function to find most similar palette color to a given color
            const findMostSimilarPaletteColor = (targetColor) => {
                let minDistance = Infinity;
                let mostSimilar = paletteRGB[0];
                
                for (const paletteColor of paletteRGB) {
                    // Calculate Euclidean distance in RGB space
                    const dr = targetColor.x - paletteColor.x;
                    const dg = targetColor.y - paletteColor.y;
                    const db = targetColor.z - paletteColor.z;
                    const distance = Math.sqrt(dr * dr + dg * dg + db * db);
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        mostSimilar = paletteColor;
                    }
                }
                
                return mostSimilar;
            };
            
            console.log("Updating", numParticles, "particles with", this.gridTargetColors.length, "colors");
            
            // USA mode (mode 7): use closest palette color directly, no lerping
            if (mode === 7) {
                for (let i = 0; i < numParticles && i < this.gridTargetColors.length; i++) {
                    const imageColor = this.gridTargetColors[i];
                    const closestPaletteColor = findMostSimilarPaletteColor(imageColor);
                    
                    const arrayOffset = i * this.particleBuffer.structSize + colorOffset;
                    this.particleBuffer.floatArray[arrayOffset] = closestPaletteColor.x;
                    this.particleBuffer.floatArray[arrayOffset + 1] = closestPaletteColor.y;
                    this.particleBuffer.floatArray[arrayOffset + 2] = closestPaletteColor.z;
                    
                    // Debug first few particles
                    if (i < 3) {
                        console.log(`Particle ${i}: wrote closest palette color (${closestPaletteColor.x.toFixed(3)}, ${closestPaletteColor.y.toFixed(3)}, ${closestPaletteColor.z.toFixed(3)}) at offset ${arrayOffset}`);
                    }
                }
                console.log(`USA mode: assigned closest palette colors to all ${numParticles} particles`);
            } else {
                // Other image modes: use lerping logic
                let lerpedCount = 0;
                for (let i = 0; i < numParticles && i < this.gridTargetColors.length; i++) {
                    let color = this.gridTargetColors[i];
                    
                    // 20% chance to lerp between most similar palette color and image color
                    if (Math.random() < 0.6) {
                        const mostSimilarPaletteColor = findMostSimilarPaletteColor(color);
                        // Lerp at a random point between most similar palette and image color
                        // Use Gaussian-like distribution weighted toward 0 (palette color)
                        // Box-Muller transform for Gaussian, then normalize and clamp to 0-1
                        const u1 = Math.random();
                        const u2 = Math.random();
                        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                        // Normalize to roughly 0-1 range, weighted heavily toward 0
                        // Use smaller std dev and lower mean to bias toward palette color
                        let lerpFactor = Math.abs(z0 * 0.25 + 0.1); // Center around 0.1, std dev 0.25, use abs to bias toward 0
                        lerpFactor = Math.max(0, Math.min(1, lerpFactor)); // Clamp to 0-1
                        const oneMinusLerp = 1.0 - lerpFactor;
                        color = new THREE.Vector3(
                            mostSimilarPaletteColor.x * lerpFactor + color.x * oneMinusLerp,
                            mostSimilarPaletteColor.y * lerpFactor + color.y * oneMinusLerp,
                            mostSimilarPaletteColor.z * lerpFactor + color.z * oneMinusLerp
                        );
                        lerpedCount++;
                    }
                    
                    const arrayOffset = i * this.particleBuffer.structSize + colorOffset;
                    this.particleBuffer.floatArray[arrayOffset] = color.x;
                    this.particleBuffer.floatArray[arrayOffset + 1] = color.y;
                    this.particleBuffer.floatArray[arrayOffset + 2] = color.z;
                    
                    // Debug first few particles
                    if (i < 3) {
                        console.log(`Particle ${i}: wrote color (${color.x.toFixed(3)}, ${color.y.toFixed(3)}, ${color.z.toFixed(3)}) at offset ${arrayOffset}`);
                        console.log(`  Buffer values: [${this.particleBuffer.floatArray[arrayOffset].toFixed(3)}, ${this.particleBuffer.floatArray[arrayOffset + 1].toFixed(3)}, ${this.particleBuffer.floatArray[arrayOffset + 2].toFixed(3)}]`);
                    }
                }
                if (lerpedCount > 0) {
                    console.log(`Lerped ${lerpedCount} particles (${(lerpedCount/numParticles*100).toFixed(1)}%) between palette and image colors`);
                }
            }
        } else {
            // Reset image mode when switching back to chaos or front gravity
            if (mode !== 3) {
                this.currentImageMode = null;
            }
            
            // Two color sphere mode: assign either #317285 or #79C8D6 to each particle
            if (twoColorSphereEnabled) {
                // Convert hex colors to RGB
                const color1 = "#317285"; // Dark blue-green
                const color2 = "#79C8D6"; // Light cyan
                const r1 = parseInt(color1.slice(1, 3), 16) / 255;
                const g1 = parseInt(color1.slice(3, 5), 16) / 255;
                const b1 = parseInt(color1.slice(5, 7), 16) / 255;
                const r2 = parseInt(color2.slice(1, 3), 16) / 255;
                const g2 = parseInt(color2.slice(3, 5), 16) / 255;
                const b2 = parseInt(color2.slice(5, 7), 16) / 255;
                
                for (let i = 0; i < numParticles; i++) {
                    // Randomly assign one of the two colors to each particle
                    const useColor1 = Math.random() < 0.5;
                    const arrayOffset = i * this.particleBuffer.structSize + colorOffset;
                    if (useColor1) {
                        this.particleBuffer.floatArray[arrayOffset] = r1;
                        this.particleBuffer.floatArray[arrayOffset + 1] = g1;
                        this.particleBuffer.floatArray[arrayOffset + 2] = b1;
                    } else {
                        this.particleBuffer.floatArray[arrayOffset] = r2;
                        this.particleBuffer.floatArray[arrayOffset + 1] = g2;
                        this.particleBuffer.floatArray[arrayOffset + 2] = b2;
                    }
                }
                console.log(`Two color sphere mode: assigned colors #317285 and #79C8D6 to all ${numParticles} particles`);
            } else {
                // Reassign colors from palette (cycling through like in init)
                const palette = [
                    "#FF56B7", "#BC96F9", "#C5F0F2", "#007487", "#007487",
                    "#153943", "#232356", "#50121A", "#FF6425", "#D6F499",
                    "#5CCAD8", "#5CCAD8", "#C5F0F2",
                ];
                const paletteRGB = palette.map(hex => {
                    const r = parseInt(hex.slice(1, 3), 16) / 255;
                    const g = parseInt(hex.slice(3, 5), 16) / 255;
                    const b = parseInt(hex.slice(5, 7), 16) / 255;
                    return new THREE.Vector3(r, g, b);
                });
                
                // Write directly to floatArray to ensure changes are reflected
                for (let i = 0; i < numParticles; i++) {
                    const colorIndex = i % paletteRGB.length;
                    const color = paletteRGB[colorIndex];
                    const arrayOffset = i * this.particleBuffer.structSize + colorOffset;
                    this.particleBuffer.floatArray[arrayOffset] = color.x;
                    this.particleBuffer.floatArray[arrayOffset + 1] = color.y;
                    this.particleBuffer.floatArray[arrayOffset + 2] = color.z;
                }
            }
        }
        
        // Always recreate the buffer to ensure a fresh reference
        // The instancedArray should sync with floatArray, but recreating ensures
        // the shader nodes get a fresh reference
        const oldBuffer = this.particleBuffer.buffer;
        this.particleBuffer.buffer = instancedArray(this.particleBuffer.floatArray, this.particleBuffer.structNode).label("particleData");
        
        console.log("Recreated buffer reference");
        console.log("Old buffer UUID:", oldBuffer?.uuid);
        console.log("New buffer UUID:", this.particleBuffer.buffer?.uuid);
        
        // Also try to update the GPU buffer directly if possible
        if (this.renderer && this.renderer.backend) {
            try {
                const bufferNode = this.particleBuffer.buffer;
                const storage = bufferNode.storage || bufferNode._storage;
                if (storage && storage.buffer) {
                    const device = this.renderer.backend.device;
                    device.queue.writeBuffer(
                        storage.buffer,
                        0,
                        this.particleBuffer.floatArray.buffer,
                        0,
                        this.particleBuffer.floatArray.byteLength
                    );
                    console.log("Also updated GPU buffer directly through device.queue.writeBuffer");
                }
            } catch (e) {
                console.log("Note: Could not update buffer directly (this is OK):", e.message);
            }
        }
        
        console.log("Verifying first particle color:", 
            this.particleBuffer.floatArray[colorOffset].toFixed(3),
            this.particleBuffer.floatArray[colorOffset + 1].toFixed(3),
            this.particleBuffer.floatArray[colorOffset + 2].toFixed(3));
        
        // Update the material's colorNode to use the new/updated buffer reference
        // This is critical - we need to get a fresh reference to the particle buffer element
        // with the new buffer, and force the material to rebuild its shader
        if (this.particleRenderer) {
            const material = this.particleRenderer.material;
            
            // Get a fresh particle reference with the new buffer
            const particle = this.particleBuffer.element(instanceIndex);
            const newColorNode = particle.get("color");
            
            // Update the colorNode
            material.colorNode = newColorNode;
            
            // Force shader rebuild by invalidating the material
            // Try multiple approaches to ensure it works
            try {
                // Invalidate the material version to force rebuild
                if (material.version !== undefined) {
                    material.version++;
                }
                
                // Clear any cached program
                if (material.program) {
                    material.program = null;
                }
                if (material._program) {
                    material._program = null;
                }
                
                // Try to invalidate node builder
                if (material.nodeBuilder) {
                    material.nodeBuilder.needsUpdate = true;
                    if (material.nodeBuilder.cache) {
                        material.nodeBuilder.cache.clear();
                    }
                }
                
                // Force material to mark as needing update
                material.needsUpdate = true;
                
                console.log("Updated material colorNode and forced shader rebuild");
                console.log("Material version:", material.version);
                console.log("New colorNode:", newColorNode);
                console.log("Buffer reference in colorNode:", newColorNode?.buffer || newColorNode?.storage || "not found");
            } catch (e) {
                console.error("Error forcing shader rebuild:", e);
            }
        }
    }

    async update(interval, elapsed) {
        const { particles, run, noise, gridNoiseStrength, dynamicViscosity, stiffness, restDensity, speed, gravity, gravitySensorReading, accelerometerReading, cursorInteraction, hidePercentage } = conf;
        
        // Update color spheres boost timer
        if (this.uniforms.colorSpheresMode.value === 1 && this.colorSpheresModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.colorSpheresModeStartTime) / 1000; // Convert to seconds
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.colorSpheresBoostTime.value = remainingBoost;
        }
        
        // Update population segment pink boost timer
        if (this.uniforms.populationSegmentPinkMode.value === 1 && this.populationSegmentPinkModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentPinkModeStartTime) / 1000; // Convert to seconds
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentPinkBoostTime.value = remainingBoost;
        }
        
        // Update population segment purple boost timer
        if (this.uniforms.populationSegmentPurpleMode.value === 1 && this.populationSegmentPurpleModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentPurpleModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentPurpleBoostTime.value = remainingBoost;
        }
        
        // Update population segment cyan boost timer
        if (this.uniforms.populationSegmentCyanMode.value === 1 && this.populationSegmentCyanModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentCyanModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentCyanBoostTime.value = remainingBoost;
        }
        
        // Update population segment teal boost timer
        if (this.uniforms.populationSegmentTealMode.value === 1 && this.populationSegmentTealModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentTealModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentTealBoostTime.value = remainingBoost;
        }
        
        // Update population segment dark blue boost timer
        if (this.uniforms.populationSegmentDarkBlueMode.value === 1 && this.populationSegmentDarkBlueModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentDarkBlueModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentDarkBlueBoostTime.value = remainingBoost;
        }
        
        // Update population segment dark purple boost timer
        if (this.uniforms.populationSegmentDarkPurpleMode.value === 1 && this.populationSegmentDarkPurpleModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentDarkPurpleModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentDarkPurpleBoostTime.value = remainingBoost;
        }
        
        // Update population segment dark red boost timer
        if (this.uniforms.populationSegmentDarkRedMode.value === 1 && this.populationSegmentDarkRedModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentDarkRedModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentDarkRedBoostTime.value = remainingBoost;
        }
        
        // Update population segment orange boost timer
        if (this.uniforms.populationSegmentOrangeMode.value === 1 && this.populationSegmentOrangeModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentOrangeModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentOrangeBoostTime.value = remainingBoost;
        }
        
        // Update population segment light green boost timer
        if (this.uniforms.populationSegmentLightGreenMode.value === 1 && this.populationSegmentLightGreenModeStartTime > 0) {
            const elapsedSinceStart = (Date.now() - this.populationSegmentLightGreenModeStartTime) / 1000;
            const remainingBoost = Math.max(0, mlsMpmSimulator.COLOR_SPHERES_BOOST_DURATION - elapsedSinceStart);
            this.uniforms.populationSegmentLightGreenBoostTime.value = remainingBoost;
        }

        this.uniforms.noise.value = noise;
        this.uniforms.gridNoiseStrength.value = gridNoiseStrength;
        this.uniforms.stiffness.value = stiffness;
        this.uniforms.gravityType.value = gravity;
				this.uniforms.gravity.value.set(0,0,0);
        this.uniforms.cursorInteraction.value = cursorInteraction ? 1 : 0;
        this.uniforms.hidePercentage.value = hidePercentage;
        // if (gravity === 0) {
        //     this.uniforms.gravity.value.set(0,0,0.2);
        // } else if (gravity === 1) {
        //     this.uniforms.gravity.value.set(0,-0.2,0);
        // } else if (gravity === 3) {
        //     this.uniforms.gravity.value.copy(gravitySensorReading).add(accelerometerReading);
        // }
        this.uniforms.dynamicViscosity.value = dynamicViscosity;
        this.uniforms.restDensity.value = restDensity;

        if (particles !== this.numParticles) {
            this.numParticles = particles;
            this.uniforms.numParticles.value = particles;
            this.kernels.p2g1.count = particles;
            this.kernels.p2g1.updateDispatchCount();
            this.kernels.p2g2.count = particles;
            this.kernels.p2g2.updateDispatchCount();
            this.kernels.g2p.count = particles;
            this.kernels.g2p.updateDispatchCount();
        }

        interval = Math.min(interval, 1/60);
        // Use speed 4.0 in front gravity mode (or cylinder mode), otherwise use configured speed
        const effectiveSpeed = (this.uniforms.frontGravityMode.value === 1) ? 4.0 : speed;
        const dt = interval * 6 * effectiveSpeed;
        this.uniforms.dt.value = dt;

        this.mousePosArray.push(this.mousePos.clone())
        if (this.mousePosArray.length > 3) { this.mousePosArray.shift(); }
        if (this.mousePosArray.length > 1) {
            this.uniforms.mouseForce.value.copy(this.mousePosArray[this.mousePosArray.length - 1]).sub(this.mousePosArray[0]).divideScalar(this.mousePosArray.length);
        }


        if (run) {
            const kernels = [this.kernels.clearGrid, this.kernels.p2g1, this.kernels.p2g2, this.kernels.updateGrid, this.kernels.g2p];
            await this.renderer.computeAsync(kernels);
        }
    }
}

export default mlsMpmSimulator;