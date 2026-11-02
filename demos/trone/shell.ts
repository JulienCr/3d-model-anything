import * as THREE from 'three';
import { box, boxUV, mesh } from '../../packages/room-runtime/helpers';
import { pointInPolygon, polygonEdges, polygonSurface, type Poly } from '../../packages/room-runtime/polygon-room';
import { buildWall, onWall, wallMatrix, wallPoint, type WallDef } from '../../packages/room-runtime/walls';
import geometry from './geometry.json';
import type { TroneMaterials } from './materials';

/** The LiDAR floor sits this far above RoomPlan's floor; photos and the scan are shifted down by it. */
export const FLOOR_Y = geometry.floorY;
/** Underside of the sprayed insulation (mesh: 2.16-2.20 over the whole room). */
export const HEIGHT = 2.18;
/** Bare concrete soffit east of the last beam, above the plasterboard partition (photo 133). */
const EAST_SOFFIT = { x: 7.5, y: 2.27 };
const THICKNESS = 0.25;

/**
 * Floor outline from wall-plane modes of the LiDAR mesh (faces 1.4-2.0 m high, 0.2 m bins). RoomPlan was
 * 0.2-0.35 m off on the north, east and south-east walls, so it is not used for geometry. Basement walls
 * are not square: the west and east walls lean 1.5-2.5 degrees off the room axes, as measured.
 */
export const POLY = geometry.polygon as Poly;

export const insideRoom = (x: number, z: number) => pointInPolygon(POLY, x, z);

export interface ShellWall { def: WallDef; group: THREE.Group; door?: { u0: number; u1: number } }

/** East door: frame edges ray-cast from photo 133 onto the east wall plane (z -0.80..0.22, top 2.08). */
const DOOR = { z0: -0.8, z1: 0.22, height: 2.08 };

const isEast = (a: number[], b: number[]) => Math.min(a[0], b[0]) > 8.4 && Math.abs(a[1] - b[1]) > 3;
const isAlcove = (a: number[], b: number[]) => Math.abs(a[1] - b[1]) < 0.1 && Math.min(a[1], b[1]) < -3.3;
const isBlockwork = (a: number[], b: number[]) => Math.min(a[0], b[0]) > 7.1 && Math.min(a[1], b[1]) > 2.1;

function buildDoor(def: WallDef, u0: number, u1: number, m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'door';
  const frame = 0.045, depth = 0.1;
  const leafW = u1 - u0 - 2 * frame, leafH = DOOR.height - frame;
  const strips: [number, number, number, number][] = [
    [u0 + frame / 2, DOOR.height / 2, frame, DOOR.height], [u1 - frame / 2, DOOR.height / 2, frame, DOOR.height], [(u0 + u1) / 2, DOOR.height - frame / 2, u1 - u0, frame],
  ];
  for (const [u, v, w, h] of strips) {
    const s = mesh(boxUV(new THREE.BoxGeometry(w, h, depth)), m.doorFrame);
    s.position.set(u, v, -depth / 2 + 0.02);
    g.add(s);
  }
  const leaf = mesh(new THREE.BoxGeometry(leafW, leafH, 0.04), m.doorLeaf);
  leaf.position.set((u0 + u1) / 2, leafH / 2, -0.03);
  g.add(leaf);
  // hinges on the south (u1) side, lever and key plate on the north side (photo 133)
  for (const v of [0.25, 1.05, 1.75, 1.95]) g.add(box([0.015, 0.1, 0.02], [u1 - frame - 0.005, v, -0.005], m.chrome));
  const hu = u0 + frame + 0.06;
  g.add(box([0.035, 0.24, 0.012], [hu, 1.0, -0.004], m.chrome));
  g.add(box([0.13, 0.02, 0.02], [hu + 0.05, 1.06, 0.018], m.chrome));
  g.position.copy(wallPoint(def, 0, 0, 0));
  g.quaternion.setFromRotationMatrix(wallMatrix(def));
  return g;
}

/** Pink boards (photos 0, 3, 12) screwed over the north wall's west half, up to the steel angle. */
function northBoards(def: WallDef, m: TroneMaterials) {
  const u = (x: number) => (x - def.origin.x) * def.u.x + (-2.9 - def.origin.z) * def.u.z;
  const [u0, u1] = [u(-7.62), u(-4.78)].sort((a, b) => a - b);
  const g = mesh(boxUV(new THREE.BoxGeometry(u1 - u0, 2.04, 0.013)), m.plasterboard, 'boards');
  onWall(g, def, (u0 + u1) / 2, 1.02, 0.0065);
  return g;
}

/** Thin finish panel on whichever wall runs under the world segment p..q (x/z), between heights v0 and v1. */
function finish(walls: ShellWall[], p: [number, number], q: [number, number], v0: number, v1: number, material: THREE.Material) {
  const mid = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  let best: ShellWall | undefined, bestD = Infinity;
  for (const w of walls) {
    const dx = mid[0] - w.def.origin.x, dz = mid[1] - w.def.origin.z;
    const u = dx * w.def.u.x + dz * w.def.u.z, d = Math.abs(dx * w.def.n.x + dz * w.def.n.z);
    if (u > 0 && u < w.def.length && d < bestD) { best = w; bestD = d; }
  }
  if (!best) return;
  const { def } = best;
  const u = (c: [number, number]) => (c[0] - def.origin.x) * def.u.x + (c[1] - def.origin.z) * def.u.z;
  const [u0, u1] = [u(p), u(q)].sort((a, b) => a - b);
  const panel = mesh(boxUV(new THREE.BoxGeometry(u1 - u0, v1 - v0, 0.008)), material, 'finish');
  onWall(panel, def, (u0 + u1) / 2, (v0 + v1) / 2, 0.004);
  best.group.add(panel);
}

/** Sprayed insulation wraps down the top of the walls as a soft lip (photos 24, 27, 39). */
function flockLip(def: WallDef, m: TroneMaterials) {
  // arc from -90 to 0 deg: after turning the axis along u it bulges down and into the room
  const lip = mesh(new THREE.CylinderGeometry(0.06, 0.06, def.length, 10, 1, true, 1.5 * Math.PI, Math.PI / 2), m.flock, 'flock-lip');
  lip.castShadow = false;
  const pivot = new THREE.Group();
  lip.rotation.z = Math.PI / 2;
  pivot.add(lip);
  onWall(pivot, def, def.length / 2, HEIGHT, 0);
  return pivot;
}

/** Clip a polygon to x <= cut (keep = 'left') or x >= cut (Sutherland-Hodgman on one plane). */
function clipX(poly: Poly, cut: number, keep: 'left' | 'right'): Poly {
  const inside = (p: number[]) => (keep === 'left' ? p[0] <= cut : p[0] >= cut);
  const out: Poly = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if (inside(a)) out.push(a);
    if (inside(a) !== inside(b)) {
      const t = (cut - a[0]) / (b[0] - a[0]);
      out.push([cut, a[1] + t * (b[1] - a[1])]);
    }
  }
  return out;
}

export function buildShell(m: TroneMaterials) {
  const walls: ShellWall[] = [];
  for (const [i, { a, b, dir, normal, length, ext }] of polygonEdges(POLY, THICKNESS).entries()) {
    const east = isEast(a, b), alcove = isAlcove(a, b), block = isBlockwork(a, b);
    let door: ShellWall['door'];
    if (east) {
      const u = (z: number) => (z - a[1]) / dir.z;
      const [u0, u1] = [u(DOOR.z0), u(DOOR.z1)].sort((p, q) => p - q);
      door = { u0, u1 };
    }
    const material = east || alcove ? m.plasterboard : block ? m.block : m.concrete;
    const top = Math.max(a[0], b[0]) > EAST_SOFFIT.x ? EAST_SOFFIT.y : HEIGHT;
    const def: WallDef = {
      name: `trone-${i}`,
      origin: new THREE.Vector3(a[0], 0, a[1]),
      u: dir, n: normal, length, thickness: THICKNESS, height: top,
      openings: door ? [{ kind: 'rect', u0: door.u0, u1: door.u1, v0: 0, v1: DOOR.height }] : [],
      panels: [], crown: false, ext, baseboardGaps: [[0, length]],
      material, trimMaterial: m.trim,
    };
    const group = buildWall(def);
    if (door) group.add(buildDoor(def, door.u0, door.u1, m));
    if (!east && !alcove && length > 1 && top === HEIGHT) group.add(flockLip(def, m));
    if (Math.abs(a[1] + 2.9) < 0.1 && Math.max(a[0], b[0]) < -4.7 && length > 2.5) group.add(northBoards(def, m));
    walls.push({ def, group, door });
  }

  // bare grey concrete either side of the painted stretch of the north wall, skim coat right of the door (photos 3, 133, 216)
  finish(walls, [-9.1, -2.87], [-7.62, -2.9], 0, HEIGHT, m.greyConcrete);
  finish(walls, [-4.75, -2.87], [-2.45, -2.92], 0, HEIGHT, m.greyConcrete);
  finish(walls, [8.68, 0.24], [8.71, 1.3], 0, 2.12, m.skim);
  finish(walls, [2.7, -2.97], [5.62, -2.94], 0, HEIGHT, m.greyConcrete);

  const ceiling = new THREE.Group();
  ceiling.name = 'ceiling';
  ceiling.add(polygonSurface(clipX(POLY, EAST_SOFFIT.x, 'left'), HEIGHT, m.flock, 'flock'));
  const soffit = polygonSurface(clipX(POLY, EAST_SOFFIT.x, 'right'), EAST_SOFFIT.y, m.soffit, 'soffit');
  soffit.material = m.soffit.clone();
  (soffit.material as THREE.Material).side = THREE.DoubleSide;
  ceiling.add(soffit);
  // last beam before the partition (depth-cloud sections): soffit sloping from 2.18 to 2.06, square east
  // face up to the higher soffit, spanning from the north wall onto the L pillar
  const beamShape = new THREE.Shape([[7.0, HEIGHT], [7.45, 2.06], [EAST_SOFFIT.x, 2.06], [EAST_SOFFIT.x, EAST_SOFFIT.y], [7.0, EAST_SOFFIT.y]].map(([x, y]) => new THREE.Vector2(x, y)));
  const beam = mesh(new THREE.ExtrudeGeometry(beamShape, { depth: 0.61 + 3.0, bevelEnabled: false }), m.flock, 'beam');
  beam.position.z = -3.0;
  ceiling.add(beam);

  // plasterboard box along the south-east wall that takes the N5 duct (depth-cloud bottom 1.92-1.94; photos 117, 170)
  ceiling.add(box([7.25 - 6.3, 2.27 - 1.93, 2.64 - 2.28], [(6.3 + 7.25) / 2, (1.93 + 2.27) / 2, (2.28 + 2.64) / 2], m.skim));
  ceiling.add(box([8.76 - 7.25, 2.27 - 1.93, 2.23 - 1.86], [(7.25 + 8.76) / 2, (1.93 + 2.27) / 2, (1.86 + 2.23) / 2], m.skim));
  // box round the N4 heating pipes east of the pillar (flat underside at 2.14 in the depth data; photo 117)
  ceiling.add(box([8.76 - 7.66, 2.27 - 2.14, 1.32 - 0.85], [(7.66 + 8.76) / 2, (2.14 + 2.27) / 2, (0.85 + 1.32) / 2], m.skim));
  // the alcove's lowered plasterboard ceiling (photos 143, 147)
  ceiling.add(box([0.62, 0.2, 0.54], [8.19, 2.08, -3.23], m.soffit));
  return { walls, floor: polygonSurface(POLY, 0, m.floor, 'floor'), ceiling };
}
