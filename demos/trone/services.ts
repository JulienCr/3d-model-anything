import * as THREE from 'three';
import { box, boxUV, mesh } from '../../packages/room-runtime/helpers';
import { onWall } from '../../packages/room-runtime/walls';
import type { TroneMaterials } from './materials';
import { buildNetworks } from './networks';
import { HEIGHT, type ShellWall } from './shell';




/** LED site floodlight on a bracket (photos 4-9, 282): dark housing, bright face. */
function floodlight(m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'floodlight';
  g.add(box([0.15, 0.12, 0.045], [0, 0, 0], m.blackPlastic));
  g.add(box([0.12, 0.09, 0.005], [0, 0, 0.025], m.ledGlow));
  for (let i = -2; i <= 2; i++) g.add(box([0.008, 0.11, 0.025], [i * 0.028, 0, -0.035], m.blackPlastic));
  g.add(box([0.18, 0.015, 0.015], [0, -0.07, 0], m.steel));
  // the spot light sits inside the housing: the fixture must not shadow it
  g.traverse(o => { o.castShadow = false; });
  return g;
}

/** Ventilation grille: frame plus louvres, built facing +z. */
function grille(w: number, h: number, m: TroneMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'grille';
  g.add(box([w, h, 0.012], [0, 0, 0.006], m.grille));
  for (let k = 1; k < 7; k++) g.add(box([w - 0.03, 0.006, 0.01], [0, -h / 2 + (k * h) / 7, 0.016], m.steel));
  return g;
}

function mountAt(walls: ShellWall[], x: number, z: number, v: number, obj: THREE.Object3D, out = 0) {
  let best: ShellWall | undefined, bestD = Infinity, bestU = 0;
  for (const w of walls) {
    const dx = x - w.def.origin.x, dz = z - w.def.origin.z;
    const u = dx * w.def.u.x + dz * w.def.u.z;
    const d = Math.abs(dx * w.def.n.x + dz * w.def.n.z);
    if (u > -0.05 && u < w.def.length + 0.05 && d < bestD) { best = w; bestD = d; bestU = u; }
  }
  if (!best) throw new Error(`no wall near ${x}, ${z}`);
  onWall(obj, best.def, bestU, v, out);
  best.group.add(obj);
  return obj;
}

/** Wall pilaster from x0..x1 (world x along the wall) standing out to `front` (world z), full height. */
function pilaster(walls: ShellWall[], x0: number, x1: number, zWall: number, front: number, material: THREE.Material) {
  const depth = Math.abs(front - zWall) + 0.02;
  return mountAt(walls, (x0 + x1) / 2, zWall, HEIGHT / 2, box([x1 - x0, HEIGHT, depth], [0, 0, depth / 2 - 0.02], material));
}

/**
 * L-section pillar under the end of the east beam (depth-cloud plan slices, 1.0-1.9 m): south arm
 * x 7.13-7.66 / z 0.90-1.22, east arm x 7.36-7.66 / z 0.61-1.22, the north-west corner open. Its top
 * stops under the heating pipes that cross it (photo 170).
 */
export const PILLAR = { x0: 7.13, x1: 7.66, xi: 7.36, z0: 0.61, zi: 0.9, z1: 1.22, top: 2.02 };

function lPillar(m: TroneMaterials) {
  const { x0, x1, xi, z0, zi, z1, top } = PILLAR;
    // a proper rotation of (x, z): mirroring the shape instead turns every side face inside out
  const shape = new THREE.Shape([[x0, zi], [xi, zi], [xi, z0], [x1, z0], [x1, z1], [x0, z1]].map(([x, z]) => new THREE.Vector2(x, z)));
  const geo = new THREE.ExtrudeGeometry(shape, { depth: top, bevelEnabled: false });
  geo.rotateX(Math.PI / 2);
  geo.translate(0, top, 0);
  boxUV(geo);
  return mesh(geo, m.pillarConcrete, 'l-pillar');
}

/** Pilasters and wall returns from 1.0-1.4 m plan slices of the depth cloud; widths and depths to +-2 cm. */
function structure(m: TroneMaterials, walls: ShellWall[]) {
  const g = new THREE.Group();
  g.name = 'structure';
  g.add(lPillar(m));
  // north wall: stepped pilaster, a deep part then a shallow return (photo 220)
  pilaster(walls, -2.42, -2.15, -2.93, -2.70, m.pilasterFace);
  pilaster(walls, -2.15, -1.92, -2.94, -2.82, m.pilasterFace);
  // south wall pilasters, west to east (photos 62, 76, 99, 110)
  pilaster(walls, -2.13, -1.93, 3.02, 2.70, m.pilasterFace);
  pilaster(walls, -0.03, 0.17, 2.94, 2.62, m.pilasterFace);
  pilaster(walls, 2.97, 3.22, 2.8, 2.57, m.pilasterFace);
  pilaster(walls, 6.0, 6.27, 2.69, 2.47, m.pilasterFace);
  // painted pilaster strip with a notice and the timber batten beside it (photos 30, 37, 42)
  mountAt(walls, -5.33, 3.04, 1.05, box([0.22, 2.1, 0.03], [0, 0, 0.015], m.skim));
  mountAt(walls, -5.33, 3.04, 1.35, box([0.15, 0.2, 0.004], [0, 0, 0.032], m.paper));
  mountAt(walls, -5.17, 3.04, 1.05, box([0.07, 2.1, 0.03], [0, 0, 0.015], m.pine));
  return g;
}

/** Strut rail and clips of the south risers (photos 55, 62) and the steel angle where the north boards stop (photos 3, 12). */
function steelwork(m: TroneMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'steelwork';
  g.add(box([0.04, HEIGHT, 0.02], [-2.16, HEIGHT / 2, 2.7], m.steel));
  g.add(box([4.4, 0.05, 0.05], [-6.95, 2.06, -2.84], m.steel));
  return g;
}

/** Fixtures on the walls: grilles, exit sign, junction box with its coiled cable. */
function wallFixtures(m: TroneMaterials, walls: ShellWall[]) {
  // positions ray-cast from photos 19, 27 and 30
  for (const [x, z] of [[-9.06, -1.37], [-9.02, 0.45], [-8.96, 2.31], [-7.8, 3.1], [-6.33, 3.04]] as const) {
    mountAt(walls, x, z, 1.86, grille(0.14, 0.14, m));
  }
  const sign = box([0.32, 0.12, 0.02], [0, 0, 0.01], m.exitSign);
  sign.castShadow = false;
  mountAt(walls, 8.57, -0.98, 2.03, sign);
  mountAt(walls, 5.45, -2.94, 1.3, box([0.1, 0.1, 0.05], [0, 0, 0.025], m.whitePlastic));
  // white PVC trunking down the south-west corner (photos 20, 22)
  mountAt(walls, -8.9, 3.12, HEIGHT / 2, box([0.04, HEIGHT, 0.04], [0.03, 0, 0.02], m.whitePlastic));
  // pipe sleeve through the south-east blockwork, ray-cast from photo 117
  const sleeve = mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.06, 24), m.chrome, 'sleeve');
  sleeve.rotation.x = Math.PI / 2;
  mountAt(walls, 8.01, 2.2, 1.45, sleeve, 0.01);
  const coil = mesh(new THREE.TorusGeometry(0.1, 0.012, 8, 28), m.blackPipe, 'cable-coil');
  mountAt(walls, 5.75, -2.94, 1.15, coil, 0.03);
}

export function buildServices(m: TroneMaterials, walls: ShellWall[]) {
  const g = new THREE.Group();
  g.name = 'services';
  g.add(structure(m, walls), buildNetworks(m), steelwork(m));
  wallFixtures(m, walls);
  // the west floodlight hangs from the end of the N1 drain (photos 4, 5, 236)
  const nw = floodlight(m);
  nw.position.set(-8.5, 1.9, -2.3);
  nw.lookAt(-6, 0.6, 0);
  // clamped under the bend of the N2 crossing pipes (photo 282; photo 276 rules out a lower position)
  const work = floodlight(m);
  work.position.set(-0.85, 1.93, 1.3);
  work.lookAt(-4, 0.3, 0.5);
  g.add(nw, work);
  return g;
}
