import * as THREE from 'three';
import { mesh } from './helpers';

export type Poly = [number, number][];

/** Even-odd point-in-polygon test on room x/z. */
export function pointInPolygon(poly: Poly, x: number, z: number) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, zi] = poly[i], [xj, zj] = poly[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

function signedArea(poly: Poly) {
  return poly.reduce((s, [x, z], i) => s + x * poly[(i + 1) % poly.length][1] - poly[(i + 1) % poly.length][0] * z, 0) / 2;
}

/** Floor or ceiling plate filling the polygon at height y (faces up; use a DoubleSide material for a ceiling). */
export function polygonSurface(poly: Poly, y: number, material: THREE.Material, name: string) {
  // rotateX(-90deg) sends shape y to world -z, hence the negated z
  const geometry = new THREE.ShapeGeometry(new THREE.Shape(poly.map(([x, z]) => new THREE.Vector2(x, -z))));
  geometry.rotateX(-Math.PI / 2);
  const m = mesh(geometry, material, name);
  m.position.y = y;
  m.castShadow = false;
  return m;
}

export interface PolygonEdge {
  /** start/end oriented so that u = dir runs left to right seen from inside */
  a: [number, number];
  b: [number, number];
  dir: THREE.Vector3;
  /** into the room */
  normal: THREE.Vector3;
  length: number;
  /** slab overrun at [start, end]: the thickness at convex corners, 0 at reflex ones */
  ext: [number, number];
}

/** One wall edge per polygon side, oriented inward, with corner overruns that never poke into the room. */
export function polygonEdges(poly: Poly, thickness: number): PolygonEdge[] {
  const area = signedArea(poly);
  const n = poly.length;
  const turn = (p: number[], q: number[], r: number[]) => (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
  return poly.map((_, i) => {
    let a = poly[i], b = poly[(i + 1) % n];
    const prev = poly[(i + n - 1) % n], next = poly[(i + 2) % n];
    let ext: [number, number] = [turn(prev, a, b) * area > 0 ? thickness : 0, turn(a, b, next) * area > 0 ? thickness : 0];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const dir = new THREE.Vector3(b[0] - a[0], 0, b[1] - a[1]).normalize();
    let normal = new THREE.Vector3(-dir.z, 0, dir.x);
    if (!pointInPolygon(poly, mid[0] + normal.x * 0.1, mid[1] + normal.z * 0.1)) {
      [a, b] = [b, a];
      ext = [ext[1], ext[0]];
      dir.negate();
      normal = new THREE.Vector3(-dir.z, 0, dir.x);
    }
    return { a, b, dir, normal, length: Math.hypot(b[0] - a[0], b[1] - a[1]), ext };
  });
}
