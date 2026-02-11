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

class mlsMpmSimulator {
    renderer = null;
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
        this.uniforms.cylinderRadius = uniform(28.0); // Radius of the repulsive cylinder
        this.uniforms.cylinderCenterX = uniform(64.0); // Center X of cylinder (center of box width)
        this.uniforms.cylinderCenterY = uniform(32.0); // Center Y of cylinder (center of box height)
        this.uniforms.gridWidth = uniform(0, "uint");
        this.uniforms.gridHeight = uniform(0, "uint");
        this.uniforms.gridStartX = uniform(0);
        this.uniforms.gridStartY = uniform(0);
        this.uniforms.gridSpacingX = uniform(0);
        this.uniforms.gridSpacingY = uniform(0);
        this.uniforms.gridZ = uniform(32);
        this.uniforms.gridRandomOffset = uniform(.5); // Random offset amount (as fraction of spacing)

        this.uniforms.mouseRayDirection = uniform(new THREE.Vector3());
        this.uniforms.mouseRayOrigin = uniform(new THREE.Vector3());
        this.uniforms.mouseForce = uniform(new THREE.Vector3());
        this.uniforms.cursorInteraction = uniform(0, "uint"); // 0 = off, 1 = on

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
            
            // Grid mode: apply force towards target grid position
            If(this.uniforms.gridMode.equal(uint(1)), () => {
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
                const targetPos = vec3(targetX, targetY, this.uniforms.gridZ);
                const toTarget = targetPos.sub(particlePosition);
                const dist = toTarget.length();
                // Apply force proportional to distance - no max speed limit in grid mode
                const forceStrength = dist.mul(10.0); // Increased multiplier, no clamp for unlimited speed
                particleVelocity.addAssign(toTarget.normalize().mul(forceStrength).mul(this.uniforms.dt));
            }).Else(() => {
                // Normal fluid simulation forces or front gravity
                If(this.uniforms.frontGravityMode.equal(uint(1)), () => {
                    // Front gravity mode: pull toward front (negative Z direction)
                    const frontGravityDir = vec3(0, 0, -1).toConst(); // Negative Z is front
                    particleVelocity.addAssign(frontGravityDir.mul(0.1).mul(this.uniforms.dt));
                }).Else(() => {
                    // Normal center gravity
                    const pn = particlePosition.div(vec3(this.uniforms.gridSize.sub(1))).sub(0.5).normalize().toConst();
                    particleVelocity.subAssign(pn.mul(0.1).mul(this.uniforms.dt));
                });
            });
            
            // Apply cylinder repulsion force when in cylinder mode
            If(this.uniforms.cylinderMode.equal(uint(1)), () => {
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


            // Only apply noise when not in grid mode and not in front gravity mode (or cylinder mode)
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
            // Only apply fluid simulation forces when not in grid mode (front gravity mode keeps them)
            If(this.uniforms.gridMode.equal(uint(0)), () => {
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

            // Only apply mouse force when not in grid mode (front gravity mode keeps it) and cursor interaction is enabled
            If(this.uniforms.gridMode.equal(uint(0)).and(this.uniforms.cursorInteraction.equal(uint(1))), () => {
                const dist = cross(this.uniforms.mouseRayDirection, particlePosition.mul(vec3(1,1,0.4)).sub(this.uniforms.mouseRayOrigin)).length()
                const force = dist.mul(0.1).oneMinus().max(0.0).pow(2);
                particleVelocity.addAssign(this.uniforms.mouseForce.mul(1).mul(force));
                particleVelocity.mulAssign(particleMass); // to ensure difference between particles
            });

            this.particleBuffer.element(instanceIndex).get('C').assign(B.mul(4));
            
            // Apply damping in front gravity mode (or cylinder mode) to reduce inertia
            If(this.uniforms.frontGravityMode.equal(uint(1)), () => {
                const dampingFactor = float(0.98).toConst(); // Reduce velocity by 2% each frame
                particleVelocity.mulAssign(dampingFactor);
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
                const numParticles = conf.particles;
                const aspectRatio = img.width / img.height;
                this.gridHeight = Math.floor(Math.sqrt(numParticles / aspectRatio));
                this.gridWidth = Math.floor(this.gridHeight * aspectRatio);
                console.log("Grid dimensions:", this.gridWidth, "x", this.gridHeight, "for", numParticles, "particles");
                
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
                    
                    // Calculate target position in grid space
                    const targetX = this.gridStartX + gridX * this.gridSpacingX;
                    const targetY = this.gridStartY + gridY * this.gridSpacingY;
                    this.gridTargetPositions.push(new THREE.Vector3(targetX, targetY, zPos));
                    
                    // Sample pixel color from image
                    // Flip both X and Y coordinates to match the correct orientation
                    const imgX = Math.floor((1.0 - (gridX / this.gridWidth)) * img.width);
                    const imgY = Math.floor((1.0 - (gridY / this.gridHeight)) * img.height);
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

    async toggleGridMode(mode) {
        // mode: 0 = chaos, 1 = canvasser, 2 = protest, 3 = front gravity, 4 = front gravity + cylinder
        const enabled = mode > 0 && mode < 3; // Only modes 1 and 2 are grid modes
        const frontGravityEnabled = mode === 3 || mode === 4;
        const cylinderEnabled = mode === 4;
        this.gridMode = enabled;
        this.uniforms.gridMode.value = enabled ? 1 : 0;
        this.uniforms.frontGravityMode.value = frontGravityEnabled ? 1 : 0;
        this.uniforms.cylinderMode.value = cylinderEnabled ? 1 : 0;
        
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
        } else {
            // Reset image mode when switching back to chaos or front gravity
            if (mode !== 3) {
                this.currentImageMode = null;
            }
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
        if (window.app && window.app.particleRenderer) {
            const material = window.app.particleRenderer.material;
            
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
        const { particles, run, noise, dynamicViscosity, stiffness, restDensity, speed, gravity, gravitySensorReading, accelerometerReading, cursorInteraction } = conf;

        this.uniforms.noise.value = noise;
        this.uniforms.stiffness.value = stiffness;
        this.uniforms.gravityType.value = gravity;
				this.uniforms.gravity.value.set(0,0,0);
        this.uniforms.cursorInteraction.value = cursorInteraction ? 1 : 0;
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