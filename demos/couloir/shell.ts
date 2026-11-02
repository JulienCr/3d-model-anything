import * as THREE from 'three';
import { buildWall, wallMatrix, wallPoint, type WallDef } from '../../packages/room-runtime/walls';
import { box, boxUV, mesh } from '../../packages/room-runtime/helpers';
import type { CouloirMaterials } from './materials';

const THICKNESS = 0.2;

/**
 * Floor outline measured on the LiDAR mesh (vertical-surface density, 1 cm bins): a main corridor
 * along x, a toilet lobby to the north-east, a beamed room to the south-west and a boxes niche at
 * the west end. RoomPlan was within 2-8 cm but missed the pillar (edges 8-10) and the door recesses.
 */
export const POLY: [number, number][] = [
  [-4.68, -0.905], [3.33, -0.905], [3.33, -3.215], [4.67, -3.215], [4.67, 0.605], [-0.75, 0.605],
  [-0.75, 3.215], [-3.34, 3.215], [-3.34, 1.44], [-3.095, 1.44], [-3.095, 1.095], [-3.69, 1.095],
  [-3.69, -0.195], [-4.68, -0.195],
];

/** Wall height per edge: up to the slab or panel ceiling above each run. */
const EDGE_HEIGHT = [2.4, 2.47, 2.47, 2.47, 2.3, 2.52, 2.52, 2.52, 2.42, 2.42, 2.3, 2.4, 2.4, 2.4];
/** Finish per edge: painted plasterboard, cream lobby paint, block infill or raw concrete. */
const EDGE_FINISH: ('paint' | 'warm' | 'blocks' | 'concrete')[] = [
  'paint', 'warm', 'warm', 'paint', 'paint', 'paint', 'paint', 'blocks', 'concrete', 'concrete', 'concrete', 'concrete', 'concrete', 'blocks',
];

export interface OpeningSpec {
  edge: number;
  /** world x/z of the opening's centre on the wall */
  at: [number, number];
  width: number;
  height: number;
  /** leaf colour / construction */
  style: 'mauve' | 'black' | 'white';
  /** how far the closed leaf sits behind the wall face (m) */
  recess?: number;
  hinge: 'start' | 'end';
  closer?: boolean;
  /** leaf swung away from the corridor by this angle (rad); a dark room shows behind */
  open?: number;
}

export const OPENINGS: OpeningSpec[] = [
  { edge: 0, at: [-2.19, -0.905], width: 0.94, height: 2.02, style: 'white', recess: 0.24, hinge: 'start', closer: true, open: 1.5 },
  { edge: 1, at: [3.33, -2.62], width: 0.84, height: 2.07, style: 'mauve', hinge: 'end' },
  { edge: 2, at: [3.92, -3.215], width: 0.94, height: 2.1, style: 'mauve', hinge: 'end' },
  { edge: 11, at: [-3.69, 0.545], width: 0.95, height: 2.07, style: 'black', recess: 0.08, hinge: 'end' },
];

export function insideRoom(x: number, z: number) {
  let inside = false;
  for (let i = 0, j = POLY.length - 1; i < POLY.length; j = i++) {
    const [xi, zi] = POLY[i], [xj, zj] = POLY[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

function signedArea() {
  return POLY.reduce((s, [x, z], i) => s + x * POLY[(i + 1) % POLY.length][1] - POLY[(i + 1) % POLY.length][0] * z, 0) / 2;
}

/** Wall-local placement: +x along the wall, +y up, +z into the room. */
export function onWall<T extends THREE.Object3D>(obj: T, def: WallDef, u: number, v: number, out = 0) {
  obj.quaternion.setFromRotationMatrix(wallMatrix(def));
  obj.position.copy(wallPoint(def, u, v, out));
  return obj;
}

function floorSurface(material: THREE.Material) {
  // rotateX(-90deg) sends shape y to world -z, hence the negated z
  const geometry = new THREE.ShapeGeometry(new THREE.Shape(POLY.map(([x, z]) => new THREE.Vector2(x, -z))));
  geometry.rotateX(-Math.PI / 2);
  const m = mesh(geometry, material, 'floor');
  m.position.y = 0.015;
  m.castShadow = false;
  return m;
}

export interface ShellWall { def: WallDef; group: THREE.Group; specs: (OpeningSpec & { u0: number; u1: number })[] }

function buildDoor(def: WallDef, o: OpeningSpec & { u0: number; u1: number }, m: CouloirMaterials) {
  const g = new THREE.Group();
  g.name = `door-${o.style}`;
  const recess = o.recess ?? 0;
  const frame = o.style === 'black' ? m.steelFrame : o.style === 'mauve' ? m.mauveFrame : m.whiteDoor;
  const leafMat = o.style === 'black' ? m.blackDoor : o.style === 'mauve' ? m.mauve : m.whiteDoor;
  const f = o.style === 'black' ? 0.05 : 0.045;
  const w = o.u1 - o.u0 - 2 * f, h = o.height - f;
  // frame: jambs and head, set flush with the leaf face
  for (const [u, v, sw, sh] of [[o.u0 + f / 2, o.height / 2, f, o.height], [o.u1 - f / 2, o.height / 2, f, o.height], [(o.u0 + o.u1) / 2, o.height - f / 2, o.u1 - o.u0, f]] as const) {
    g.add(box([sw, sh, 0.06], [u, v, -recess + 0.012], frame));
  }
  // leaf, handle and closer body share a pivot on the hinge edge
  const hingeU = o.hinge === 'start' ? o.u0 + f : o.u1 - f;
  const dir = o.hinge === 'start' ? 1 : -1;
  const pivot = new THREE.Group();
  pivot.position.set(hingeU, 0, -recess - 0.01);
  if (o.open) pivot.rotation.y = dir * o.open;
  const leaf = mesh(boxUV(new THREE.BoxGeometry(w, h, 0.045)), leafMat, 'leaf');
  leaf.position.set(dir * w / 2, h / 2 + 0.008, 0);
  const free = dir * (w - 0.07);
  pivot.add(leaf, box([0.04, 0.24, 0.008], [free, 1.0, 0.027], m.chrome));
  pivot.add(box([0.13, 0.02, 0.02], [free - dir * 0.06, 1.06, 0.06], m.chrome), box([0.02, 0.02, 0.05], [free, 1.06, 0.045], m.chrome));
  if (o.closer) pivot.add(box([0.27, 0.06, 0.05], [dir * 0.2, h - 0.06, 0.05], m.lampBody));
  if (o.style === 'white') {
    const plaque = mesh(new THREE.PlaneGeometry(0.16, 0.04), m.signInterdit);
    plaque.position.set(dir * w / 2, 1.62, 0.024);
    pivot.add(plaque);
  }
  g.add(pivot);
  if (o.open) {
    // unlit room behind the open door (scanned floor reaches 2.3 m beyond; photos 140, 166)
    const room = mesh(new THREE.BoxGeometry(1.8, 2.3, 2.3), m.void, 'beyond');
    room.position.set((o.u0 + o.u1) / 2, 1.15, -recess - 1.16);
    room.castShadow = room.receiveShadow = false;
    g.add(room);
  }
  if (recess > 0.02) {
    // painted reveals between the wall face and the recessed frame
    for (const u of [o.u0, o.u1]) g.add(box([0.01, o.height, recess], [u, o.height / 2, -recess / 2], m.reveal));
    g.add(box([o.u1 - o.u0, 0.01, recess], [(o.u0 + o.u1) / 2, o.height, -recess / 2], m.reveal));
    g.add(box([o.u1 - o.u0, 0.015, recess + 0.01], [(o.u0 + o.u1) / 2, 0.0075, -recess / 2], m.floor));
  }
  g.position.copy(wallPoint(def, 0, 0, 0));
  g.quaternion.setFromRotationMatrix(wallMatrix(def));
  return g;
}

export function buildShell(m: CouloirMaterials) {
  const area = signedArea();
  const n = POLY.length;
  const walls: ShellWall[] = [];
  for (let i = 0; i < n; i++) {
    let a = POLY[i], b = POLY[(i + 1) % n];
    const prev = POLY[(i + n - 1) % n], next = POLY[(i + 2) % n];
    const turn = (p: number[], q: number[], r: number[]) => (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
    // convex corners overlap by the slab thickness; at reflex corners the end cap stops 2 mm short,
    // or it is coplanar with the next wall's face and z-fights
    const REFLEX = -0.002;
    let ext: [number, number] = [turn(prev, a, b) * area > 0 ? THICKNESS : REFLEX, turn(a, b, next) * area > 0 ? THICKNESS : REFLEX];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const dir = new THREE.Vector3(b[0] - a[0], 0, b[1] - a[1]).normalize();
    let normal = new THREE.Vector3(-dir.z, 0, dir.x);
    if (!insideRoom(mid[0] + normal.x * 0.1, mid[1] + normal.z * 0.1)) {
      [a, b] = [b, a];
      ext = [ext[1], ext[0]];
      dir.negate();
      normal = new THREE.Vector3(-dir.z, 0, dir.x);
    }
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const specs = OPENINGS.filter(o => o.edge === i).map(o => {
      const c = ((o.at[0] - a[0]) * dir.x + (o.at[1] - a[1]) * dir.z);
      return { ...o, u0: c - o.width / 2, u1: c + o.width / 2 };
    });
    const def: WallDef = {
      name: `couloir-${i}`,
      origin: new THREE.Vector3(a[0], 0, a[1]),
      u: dir, n: normal, length, thickness: THICKNESS, height: EDGE_HEIGHT[i],
      openings: specs.map(s => ({ kind: 'rect' as const, u0: s.u0, u1: s.u1, v0: 0, v1: s.height })),
      panels: [], crown: false, ext,
      baseboardGaps: [[0, length]],
      material: { paint: m.paint, warm: m.paintWarm, blocks: m.blocks, concrete: m.concrete }[EDGE_FINISH[i]], trimMaterial: m.reveal,
    };
    const group = buildWall(def);
    for (const s of specs) group.add(buildDoor(def, s, m));
    if (EDGE_FINISH[i] === 'paint' || EDGE_FINISH[i] === 'warm') {
      // red-brown sealant bead where the plasterboard meets the slab (photos 59-61, 153)
      let start = 0;
      for (const [g0, g1] of [...specs.map(s => [s.u0, s.u1]).sort((p, q) => p[0] - q[0]), [length, length]]) {
        if (g0 - start > 0.02) group.add(onWall(box([g0 - start, 0.012, 0.006], [0, 0, 0], m.sealant), def, (start + g0) / 2, 0.021, 0.003));
        start = g1;
      }
    }
    walls.push({ def, group, specs });
  }
  return { walls, floor: floorSurface(m.floor), ceiling: new THREE.Group() };
}
