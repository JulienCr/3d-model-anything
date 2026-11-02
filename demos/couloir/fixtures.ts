import * as THREE from 'three';
import { box, cylinder, mesh } from '../../packages/room-runtime/helpers';
import { mulberry32 } from '../../packages/room-runtime/materials';
import type { CouloirMaterials } from './materials';
import { onWall, type ShellWall } from './shell';

type Rect = [number, number, number, number];

/** Ceiling slabs [x0, x1, z0, z1, y] from the mesh's down-facing surfaces (2 cm histogram modes). */
const SLABS: [...Rect, number][] = [
  [-3.69, 4.67, -0.905, 0.605, 2.25],
  [3.33, 4.67, -1.55, -0.905, 2.25],
  [-3.69, -0.75, 0.605, 1.05, 2.25],
  [-3.34, -0.75, 1.32, 1.6, 2.4],
  [-3.34, -1.75, 1.85, 3.215, 2.49],
  [-4.68, -3.69, -0.905, -0.195, 2.365],
];
/** Wood-wool acoustic panels over the toilet lobby, 0.6 m grid. */
const WOOD_WOOL: [...Rect, number] = [3.33, 4.67, -3.215, -2.25, 2.45];
/** Downstand beams [x0, x1, z0, z1, bottom, top]. */
const BEAMS: [...Rect, number, number][] = [
  [3.33, 4.67, -2.25, -1.55, 2.09, 2.46],
  [4.5, 4.67, -3.215, -2.25, 2.08, 2.46],
  [-3.69, -3.45, -0.905, 1.095, 2.11, 2.26],
  [-3.95, -3.69, -0.905, -0.195, 2.11, 2.38],
  [-3.69, -0.75, 1.05, 1.32, 2.04, 2.41],
  [-3.34, -1.75, 1.6, 1.85, 2.23, 2.5],
  [-1.75, -0.75, 1.6, 3.215, 2.25, 2.5],
];
/** Fluorescent battens along x: centre and length, from over-exposed scan patches and photos 49, 72, 83, 85. The beamed room has none. */
export const TUBES: { at: [number, number, number]; length: number }[] = [
  { at: [2.7, 2.19, -0.17], length: 1.5 },
  { at: [-0.85, 2.19, -0.2], length: 1.5 },
  { at: [-2.55, 2.19, -0.2], length: 1.5 },
];

function slab(x0: number, x1: number, z0: number, z1: number, y: number, material: THREE.Material) {
  const s = box([x1 - x0, 0.03, z1 - z0], [(x0 + x1) / 2, y + 0.015, (z0 + z1) / 2], material);
  s.castShadow = false;
  return s;
}

function pipe(radius: number, from: THREE.Vector3, to: THREE.Vector3, material: THREE.Material, segments = 20) {
  const dir = to.clone().sub(from);
  const m = mesh(new THREE.CylinderGeometry(radius, radius, dir.length(), segments), material);
  m.position.copy(from).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return m;
}

/** Polyline duct with rounded bends (quadratic fillets of radius r at each corner). */
function duct(points: THREE.Vector3[], radius: number, bend: number, material: THREE.Material) {
  const path = new THREE.CurvePath<THREE.Vector3>();
  let start = points[0].clone();
  for (let i = 1; i < points.length - 1; i++) {
    const p = points[i];
    const a = p.clone().sub(points[i - 1]).normalize(), b = points[i + 1].clone().sub(p).normalize();
    const p0 = p.clone().addScaledVector(a, -bend), p1 = p.clone().addScaledVector(b, bend);
    path.add(new THREE.LineCurve3(start, p0));
    path.add(new THREE.QuadraticBezierCurve3(p0, p, p1));
    start = p1;
  }
  path.add(new THREE.LineCurve3(start, points[points.length - 1]));
  const g = new THREE.TubeGeometry(path, 160, radius, 20, false);
  return mesh(g, material, 'duct');
}

function ceilings(m: CouloirMaterials, ceiling: THREE.Group) {
  const southWing = (x0: number, z0: number) => x0 < -0.7 && z0 > 0.6;
  for (const [x0, x1, z0, z1, y] of SLABS) ceiling.add(slab(x0, x1, z0, z1, y, southWing(x0, z0) ? m.concrete : m.ceiling));
  const [x0, x1, z0, z1, y] = WOOD_WOOL;
  ceiling.add(slab(x0, x1, z0, z1, y, m.woodWool));
  for (let x = x0 + 0.6; x < x1; x += 0.6) ceiling.add(box([0.008, 0.004, z1 - z0], [x, y - 0.002, (z0 + z1) / 2], m.darkGrey));
  // small recessed downlights in the panels (photos 7-14)
  for (const [x, z] of [[3.75, -2.9], [4.15, -2.55]] as const) {
    const ring = mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.01, 20), m.darkGrey);
    const lamp = mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.012, 16), m.tube);
    ring.position.set(x, y - 0.005, z);
    lamp.position.set(x, y - 0.007, z);
    ceiling.add(ring, lamp);
  }
  for (const [bx0, bx1, bz0, bz1, b0, b1] of BEAMS) {
    const beam = box([bx1 - bx0, b1 - b0, bz1 - bz0], [(bx0 + bx1) / 2, (b0 + b1) / 2, (bz0 + bz1) / 2], southWing(bx0, bz0) ? m.concrete : m.beam);
    beam.name = 'beam';
    ceiling.add(beam);
  }
}

function tubeLight(m: CouloirMaterials, length: number) {
  const g = new THREE.Group();
  g.name = 'batten';
  const t = mesh(new THREE.CylinderGeometry(0.03, 0.03, length - 0.1, 16), m.tube);
  t.rotation.z = Math.PI / 2;
  t.castShadow = false;
  g.add(t);
  for (const s of [-1, 1]) {
    const cap = mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.06, 16), m.lampCap);
    cap.rotation.z = Math.PI / 2;
    cap.position.x = s * (length / 2 - 0.03);
    g.add(cap, box([0.02, 0.05, 0.02], [s * (length / 2 - 0.05), 0.045, 0], m.lampCap));
  }
  g.userData.physics = { type: 'none' };
  return g;
}

/** Spiral galvanised duct: from the fire damper in the south-east corner along the south wall, then diagonally into the north wall (photos 26, 31, 43, 46, 49). */
function ducts(m: CouloirMaterials, ceiling: THREE.Group) {
  const r = 0.11, y = 2.12;
  const V = (x: number, yy: number, z: number) => new THREE.Vector3(x, yy, z);
  ceiling.add(duct([V(4.42, 1.97, 0.4), V(4.1, 1.97, 0.4), V(3.75, y, 0.46), V(1.75, y, 0.46), V(1.35, y, -0.5), V(1.35, 1.97, -0.6), V(1.35, 1.97, -0.92)], r, 0.18, m.galvanised));
  // seam bands every 1.2 m along the straight runs
  const band = (p: THREE.Vector3, axis: 'x' | 'z') => {
    const b = mesh(new THREE.CylinderGeometry(r + 0.006, r + 0.006, 0.025, 20), m.galvanised);
    if (axis === 'x') b.rotation.z = Math.PI / 2; else b.rotation.x = Math.PI / 2;
    b.position.copy(p);
    ceiling.add(b);
  };
  for (let x = 1.5; x < 4.2; x += 1.2) band(V(x, y, 0.46), 'x');
  // fire damper with its red actuator housing in the south-east corner (photos 26-29, 57)
  const damper = mesh(new THREE.CylinderGeometry(r + 0.02, r + 0.02, 0.2, 20), m.red);
  damper.rotation.z = Math.PI / 2;
  damper.position.set(4.55, 1.97, 0.4);
  ceiling.add(damper, box([0.14, 0.2, 0.12], [4.5, 1.97, 0.22], m.red), box([0.1, 0.08, 0.08], [4.5, 2.1, 0.22], m.darkGrey));
  const collar = mesh(new THREE.CylinderGeometry(r + 0.03, r + 0.03, 0.03, 20), m.galvanised);
  collar.rotation.x = Math.PI / 2;
  collar.position.set(1.35, 1.97, -0.89);
  ceiling.add(collar);
  ceiling.add(box([0.12, 0.1, 0.1], [3.4, 2.13, 0.33], m.lampCap), box([0.12, 0.1, 0.1], [4.15, 2.13, 0.33], m.lampCap));
  // hangers
  for (const [x, z] of [[3.3, 0.46], [2.4, 0.46], [1.75, 0.46]] as const) ceiling.add(box([0.02, 0.03, 0.02], [x, 2.235, z], m.galvanised));
}

/** White PVC conduits along the wall heads (photos 0-4, 22, 103). */
function conduits(m: CouloirMaterials, ceiling: THREE.Group) {
  const runs: [number, number, number, number, number, number][] = [
    [-3.4, 2.2, -0.88, 3.3, 2.2, -0.88],
    [3.36, 2.42, -3.19, 3.36, 2.42, -2.25],
    [3.36, 2.42, -3.19, 4.64, 2.42, -3.19],
    [4.64, 2.2, -0.88, 4.64, 2.2, 0.0],
    [-3.66, 2.2, -0.88, -3.66, 2.2, 0.1],
  ];
  for (const [ax, ay, az, bx, by, bz] of runs) ceiling.add(pipe(0.012, new THREE.Vector3(ax, ay, az), new THREE.Vector3(bx, by, bz), m.conduit, 8));
}

function exitSign(material: THREE.Material, m: CouloirMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'exit-sign';
  g.add(box([0.34, 0.13, 0.04], [0, 0, 0.02], m.lampBody));
  const face = mesh(new THREE.PlaneGeometry(0.32, 0.115), material);
  face.position.z = 0.041;
  face.castShadow = false;
  g.add(face);
  return g;
}

function callPoint(m: CouloirMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'call-point';
  g.add(box([0.09, 0.09, 0.05], [0, 0, 0.025], m.red), box([0.05, 0.05, 0.006], [0, 0, 0.053], m.lampBody));
  return g;
}

/** Wall-mounted items: signs, call points, the evacuation plan, a bulkhead light. */
function wallItems(m: CouloirMaterials, walls: ShellWall[]) {
  const at = (edge: number, x: number, z: number, v: number, obj: THREE.Object3D, out = 0) => {
    const w = walls[edge];
    const u = (x - w.def.origin.x) * w.def.u.x + (z - w.def.origin.z) * w.def.u.z;
    onWall(obj, w.def, u, v, out);
    w.group.add(obj);
  };
  at(3, 4.67, -0.15, 2.0, exitSign(m.exitRight, m));
  at(11, -3.69, -0.08, 1.4, callPoint(m), 0.0);
  const toilets = mesh(new THREE.PlaneGeometry(0.2, 0.05), m.signToilets);
  toilets.castShadow = false;
  at(1, 3.33, -2.35, 1.6, toilets, 0.016);
  // raw concrete columns framing the block infill of the beamed room's west wall (photos 125, 147-151)
  for (const [z0, z1] of [[1.44, 1.92], [2.93, 3.215]] as const) {
    at(7, -3.34, (z0 + z1) / 2, 1.26, box([z1 - z0, 2.52, 0.02], [0, 0, 0.01], m.concrete));
  }
  // evacuation plan on the pillar's east face, in an aluminium frame
  const plan = new THREE.Group();
  plan.add(box([0.36, 0.48, 0.012], [0, 0, 0.006], m.chrome));
  const sheet = mesh(new THREE.PlaneGeometry(0.33, 0.45), m.evacPlan);
  sheet.position.z = 0.013;
  plan.add(sheet);
  at(9, -3.095, 1.27, 1.55, plan);
  // round bulkhead on the north wall where the duct enters (photos 44, 45)
  const bulk = mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.06, 28), m.lampBody);
  bulk.rotation.x = Math.PI / 2;
  const lens = mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.065, 28), m.solid('#f4f4f2', 0.3));
  lens.rotation.x = Math.PI / 2;
  const bulkhead = new THREE.Group();
  bulkhead.add(bulk, lens);
  bulkhead.children.forEach(c => (c.position.z = 0.03));
  at(0, 1.75, -0.905, 1.8, bulkhead);
}

/** Exit signs hung on the beam faces (photos 79, 87, 103, 125). */
function beamSigns(m: CouloirMaterials, ceiling: THREE.Group) {
  const s1 = exitSign(m.exitDown, m);
  s1.position.set(-3.44, 2.04, 0.5);
  s1.rotation.y = Math.PI / 2;
  ceiling.add(s1);
  // grey drain pipe dropping out of the beamed room's ceiling (photos 133, 144-146)
  ceiling.add(pipe(0.022, new THREE.Vector3(-2.33, 2.49, 2.25), new THREE.Vector3(-2.33, 2.0, 2.25), m.darkGrey, 12));
}

/** Cardboard boxes stacked in the niche under black and white pipes (photos 88-100). */
function niche(m: CouloirMaterials, group: THREE.Group) {
  const rnd = mulberry32(5);
  const [x, z] = [-4.3, -0.55];
  let y0 = 0.015;
  for (let k = 0; k < 5 && y0 < 1.6; k++) {
    const w = 0.32 + rnd() * 0.14, h = 0.28 + rnd() * 0.18, d = 0.3 + rnd() * 0.12;
    const b = box([w, h, d], [x + (rnd() - 0.5) * 0.06, y0 + h / 2, z + (rnd() - 0.5) * 0.08], rnd() < 0.3 ? m.cardboardDark : m.cardboard);
    b.rotation.y = (rnd() - 0.5) * 0.15;
    b.name = 'carton';
    b.userData.physics = { type: 'dynamic', mass: 3, shape: 'box', surface: 'cardboard' };
    group.add(b);
    y0 += h;
  }
  // insulated black flow/return pipes and a white drain across the back of the niche (photos 92-97)
  for (const [x, y, r, mat] of [[-4.58, 2.06, 0.05, m.pipeBlack], [-4.45, 2.18, 0.05, m.pipeBlack], [-4.32, 2.24, 0.03, m.pipeWhite]] as const) {
    group.add(pipe(r, new THREE.Vector3(x, y, -0.905), new THREE.Vector3(x, y, -0.195), mat));
  }
  group.add(pipe(0.02, new THREE.Vector3(-3.75, 0.1, -0.24), new THREE.Vector3(-3.75, 2.2, -0.24), m.conduit, 8));
}

/** Mops, brooms and buckets in the south-east corner of the beamed room (photos 153-158). */
function cleaningCorner(m: CouloirMaterials, group: THREE.Group) {
  const bucket = (x: number, z: number, r: number, h: number, mat: THREE.Material) => {
    const b = cylinder(r, h, [x, h / 2 + 0.015, z], mat, 24, r * 0.85);
    b.name = 'bucket';
    b.userData.physics = { type: 'dynamic', mass: 1.2, shape: 'cylinder', surface: 'plastic' };
    group.add(b);
  };
  bucket(-1.0, 2.88, 0.17, 0.3, m.darkGrey);
  bucket(-1.2, 3.05, 0.14, 0.26, m.bucketGrey);
  const crate = box([0.3, 0.22, 0.42], [-0.95, 0.125, 2.5], m.pipeBlack);
  crate.name = 'crate';
  crate.userData.physics = { type: 'dynamic', mass: 1.5, shape: 'box', surface: 'plastic' };
  group.add(crate);
  for (const [z, col] of [[2.45, '#d8302b'], [2.55, '#3a8fc4'], [2.62, '#9aa0a6'], [2.7, '#d8302b']] as const) {
    group.add(pipe(0.012, new THREE.Vector3(-1.25, 0.05, z - 0.1), new THREE.Vector3(-0.78, 1.42, z + 0.06), m.solid(col, 0.5)));
  }
}

export function buildFixtures(m: CouloirMaterials, walls: ShellWall[], ceiling: THREE.Group, group: THREE.Group) {
  ceilings(m, ceiling);
  ducts(m, ceiling);
  conduits(m, ceiling);
  beamSigns(m, ceiling);
  for (const t of TUBES) {
    const l = tubeLight(m, t.length);
    l.position.set(...t.at);
    ceiling.add(l);
  }
  wallItems(m, walls);
  niche(m, group);
  cleaningCorner(m, group);
}

export function lighting() {
  const lights = new THREE.Group();
  lights.name = 'couloir-lights';
  lights.add(new THREE.HemisphereLight('#ffffff', '#d6d2ca', 1.1));
  TUBES.forEach((t, i) => {
    const spot = new THREE.SpotLight('#f4f7ff', 9, 0, THREE.MathUtils.degToRad(85), 1, 2);
    spot.position.set(t.at[0], t.at[1] - 0.05, t.at[2]);
    spot.target.position.set(t.at[0], 0, t.at[2]);
    if (i === 0 || i === 1) {
      spot.castShadow = true;
      spot.shadow.mapSize.set(1024, 1024);
      spot.shadow.camera.near = 0.1;
      spot.shadow.camera.far = 7;
      spot.shadow.bias = -0.0006;
      spot.shadow.normalBias = 0.02;
      spot.shadow.radius = 6;
    }
    lights.add(spot, spot.target);
  });
  const lobby = new THREE.PointLight('#fff1dc', 0.9, 0, 2);
  lobby.position.set(3.95, 2.3, -2.75);
  lights.add(lobby);
  return lights;
}
