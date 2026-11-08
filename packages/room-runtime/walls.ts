import * as THREE from 'three';
import { materials } from './materials';
import { arc, boxUV, cylinder, mesh, sweep } from './helpers';

const WALL_TOP = 3.24; // Legacy default; rooms can supply WallDef.height.

export type Opening =
  | { kind: 'rect'; u0: number; u1: number; v0: number; v1: number; casing?: boolean }
  | { kind: 'arch'; u0: number; u1: number; v0: number; spring: number };

export interface WallDef {
  name: string;
  origin: THREE.Vector3; // interior face, left end, floor
  u: THREE.Vector3; // along the wall, left → right when facing it from the room
  n: THREE.Vector3; // into the room
  length: number;
  thickness: number;
  openings: Opening[];
  /** picture-frame panel moldings [u0, v0, u1, v1] */
  panels: [number, number, number, number][];
  /** extra [u0, u1] ranges without baseboard (casing openings are added automatically) */
  baseboardGaps?: [number, number][];
  /** wall height (defaults to 3.24 m) */
  height?: number;
  /** crown molding along the top (default true) */
  crown?: boolean;
  /** how far the slab runs past each end [start, end] (default 0.3 both; 0 at outside corners) */
  ext?: [number, number];
  /** paint + trim materials (default: the palette's wall/trim) */
  material?: THREE.Material;
  trimMaterial?: THREE.Material;
}

export function wallPoint(w: WallDef, u: number, v: number, out = 0) {
  return w.origin.clone().addScaledVector(w.u, u).add(new THREE.Vector3(0, v, 0)).addScaledVector(w.n, out);
}

/** Matrix mapping wall-local (u, v, out) to world */
export function wallMatrix(w: WallDef) {
  return new THREE.Matrix4().makeBasis(w.u, new THREE.Vector3(0, 1, 0), w.n).setPosition(w.origin);
}

/** Wall-local placement: +x along the wall, +y up, +z into the room. */
export function onWall<T extends THREE.Object3D>(obj: T, def: WallDef, u: number, v: number, out = 0) {
  obj.quaternion.setFromRotationMatrix(wallMatrix(def));
  obj.position.copy(wallPoint(def, u, v, out));
  return obj;
}

// --- profiles ([s, t] — s: into the molding's field, t: out from the surface) ---

const PANEL_PROFILE: [number, number][] = [
  [0, 0], [0, 0.006], [0.004, 0.011], [0.01, 0.015], [0.018, 0.016], [0.026, 0.013], [0.03, 0.009], [0.036, 0.006], [0.042, 0.002], [0.045, 0],
];
const CASING_PROFILE: [number, number][] = [
  [-0.004, 0], [-0.004, 0.026], [-0.004, 0.026], [0.006, 0.028], [0.02, 0.027], [0.024, 0.022], [0.03, 0.022], [0.07, 0.018], [0.08, 0.014], [0.09, 0.01], [0.095, 0.004], [0.095, 0],
];
const BASEBOARD_PROFILE: [number, number][] = [
  // Clockwise, like the casing profile: outward fronts, downward undersides.
  // The reversed winding exposed an upward-facing underside coplanar with the floor.
  [0, 0.162], [0.006, 0.158], [0.012, 0.15], [0.012, 0.145], [0.014, 0.138], [0.018, 0.13], [0.018, 0], [0, 0],
];
// crown hangs from the ceiling: t is negative (down), s into the room
const CROWN_PROFILE: [number, number][] = [
  [0, -0.11], [0.01, -0.11], [0.01, -0.1], [0.018, -0.095], ...arc(0.1, -0.095 + 0.0, 0.082, Math.PI, Math.PI / 2, 8).map(([s, t]): [number, number] => [s, t]), [0.105, -0.008], [0.11, 0], [0, 0],
];
export function wallShape(w: WallDef, ext = 0.3) {
  // run past the corners so slabs overlap
  const [e0, e1] = w.ext ?? [ext, ext];
  const top = w.height ?? WALL_TOP;
  const shape = new THREE.Shape();
  // Openings that reach the floor are notches in the outline (holes can't touch the edge)
  const notches = w.openings.filter((o) => o.kind === 'rect' && o.v0 === 0).sort((a, b) => a.u0 - b.u0);
  shape.moveTo(-e0, 0);
  for (const o of notches) {
    if (o.kind !== 'rect') continue;
    shape.lineTo(o.u0, 0);
    shape.lineTo(o.u0, o.v1);
    shape.lineTo(o.u1, o.v1);
    shape.lineTo(o.u1, 0);
  }
  shape.lineTo(w.length + e1, 0);
  shape.lineTo(w.length + e1, top);
  shape.lineTo(-e0, top);
  shape.closePath();
  for (const o of w.openings) {
    if (notches.includes(o)) continue;
    const hole = new THREE.Path();
    if (o.kind === 'rect') {
      hole.moveTo(o.u0, o.v0);
      hole.lineTo(o.u0, o.v1);
      hole.lineTo(o.u1, o.v1);
      hole.lineTo(o.u1, o.v0);
      hole.closePath();
    } else {
      const r = (o.u1 - o.u0) / 2;
      hole.moveTo(o.u0, o.v0);
      hole.lineTo(o.u0, o.spring);
      hole.absarc(o.u0 + r, o.spring, r, Math.PI, 0, true);
      hole.lineTo(o.u1, o.v0);
      hole.closePath();
    }
    shape.holes.push(hole);
  }
  return shape;
}

/**
 * ExtrudeGeometry puts every side face in group 1 (trim). Keep that for door and
 * window jambs, but move the slab's end caps to group 0 (paint): at outside
 * corners they're visible, and real paint wraps around the corner.
 */
function paintSlabEnds(geo: THREE.ExtrudeGeometry, w: WallDef) {
  const [e0, e1] = w.ext ?? [0.3, 0.3];
  const pos = geo.getAttribute('position');
  const paint: number[] = [], trim: number[] = [];
  for (const g of geo.groups) {
    for (let i = g.start; i < g.start + g.count; i += 3) {
      const tri = [i, i + 1, i + 2];
      let end = false;
      if (g.materialIndex === 1) {
        const xs = tri.map((k) => pos.getX(k));
        end = xs.every((x) => Math.abs(x + e0) < 1e-4) || xs.every((x) => Math.abs(x - (w.length + e1)) < 1e-4);
      }
      (g.materialIndex === 0 || end ? paint : trim).push(...tri);
    }
  }
  geo.clearGroups();
  geo.setIndex([...paint, ...trim]);
  geo.addGroup(0, paint.length, 0);
  geo.addGroup(paint.length, trim.length, 1);
  return geo;
}

export function buildWall(w: WallDef, m = materials(), ext = 0.3) {
  const group = new THREE.Group();
  group.name = `wall-${w.name}`;
  const top = w.height ?? WALL_TOP;
  const trim = w.trimMaterial ?? m.trim;

  const geo = paintSlabEnds(new THREE.ExtrudeGeometry(wallShape(w, ext), { depth: w.thickness, bevelEnabled: false, curveSegments: 24 }), w);
  geo.translate(0, 0, -w.thickness);
  const slab = mesh(geo, [w.material ?? m.wall, trim]);
  slab.applyMatrix4(wallMatrix(w));
  slab.name = 'slab';
  group.add(slab);

  const toWorld = (u: number, v: number, out = 0) => wallPoint(w, u, v, out);

  // picture-frame panels
  for (const [u0, v0, u1, v1] of w.panels) {
    const path = [toWorld(u0, v0), toWorld(u1, v0), toWorld(u1, v1), toWorld(u0, v1)];
    group.add(mesh(sweep(PANEL_PROFILE, path, w.n, { closed: true }), trim, 'panel'));
  }

  // crown + baseboard: straight runs that overlap in the corners, which reads as a miter.
  // Runs go toward -u so the profile's "left" points into the room.
  const up = new THREE.Vector3(0, 1, 0);
  if (w.crown !== false) {
    group.add(mesh(sweep(CROWN_PROFILE, [toWorld(w.length, top), toWorld(0, top)], up, { caps: false }), trim, 'crown'));
  }
  const gaps: [number, number][] = [...(w.baseboardGaps ?? [])];
  for (const o of w.openings) if (o.kind === 'rect' && o.casing && o.v0 === 0) gaps.push([o.u0 - 0.095, o.u1 + 0.095]);
  gaps.sort((a, b) => a[0] - b[0]);
  let start = 0;
  for (const [g0, g1] of [...gaps, [w.length, w.length] as [number, number]]) {
    if (g0 - start > 0.01) group.add(mesh(sweep(BASEBOARD_PROFILE, [toWorld(g0, 0), toWorld(start, 0)], up), trim, 'baseboard'));
    start = g1;
  }

  // casings
  for (const o of w.openings) {
    if (o.kind === 'rect' && o.casing && o.v0 === 0) {
      const path = [toWorld(o.u0, 0), toWorld(o.u0, o.v1), toWorld(o.u1, o.v1), toWorld(o.u1, 0)];
      group.add(mesh(sweep(CASING_PROFILE, path, w.n), trim, 'casing'));
    } else if (o.kind === 'rect' && o.casing) {
      // window: casing all the way around (path order keeps the profile's left side outward)
      const path = [toWorld(o.u0, o.v0), toWorld(o.u0, o.v1), toWorld(o.u1, o.v1), toWorld(o.u1, o.v0)];
      group.add(mesh(sweep(CASING_PROFILE, path, w.n, { closed: true }), trim, 'casing'));
    }
    if (o.kind === 'arch') group.add(buildWindow(w, o));
  }
  return group;
}

/** Arched double-hung window with transom, set into the wall reveal */
function buildWindow(w: WallDef, o: Extract<Opening, { kind: 'arch' }>, m = materials()) {
  const g = new THREE.Group();
  g.name = 'window';
  const r = (o.u1 - o.u0) / 2;
  const cu = o.u0 + r;
  const toWorld = (u: number, v: number, out = 0) => wallPoint(w, u, v, out);

  // casing around the arch, stopping on the stool
  const stool = o.v0 - 0.01;
  const archPts = arc(cu, o.spring, r, Math.PI, 0, 24).map(([u, v]) => toWorld(u, v));
  const casingPath = [toWorld(o.u0, stool), ...archPts, toWorld(o.u1, stool)];
  g.add(mesh(sweep(CASING_PROFILE, casingPath, w.n), m.trim, 'casing'));

  // stool (interior sill) + apron
  const stoolGeo = boxUV(new THREE.BoxGeometry(o.u1 - o.u0 + 0.28, 0.03, 0.2 + 0.05));
  const stoolMesh = mesh(stoolGeo, m.trim);
  stoolMesh.position.copy(toWorld(cu, stool - 0.015, 0.05 - 0.1));
  stoolMesh.quaternion.setFromRotationMatrix(wallMatrix(w));
  g.add(stoolMesh);
  const apron = mesh(boxUV(new THREE.BoxGeometry(o.u1 - o.u0 + 0.19, 0.09, 0.02)), m.trim);
  apron.position.copy(toWorld(cu, stool - 0.075, 0.01));
  apron.quaternion.copy(stoolMesh.quaternion);
  g.add(apron);

  // sash frame, set back in the reveal
  const depth = -0.12;
  const bar = 0.045;
  const frameMat = m.trim;
  const q = stoolMesh.quaternion;
  const addBar = (u: number, v: number, du: number, dv: number) => {
    const b = mesh(new THREE.BoxGeometry(du, dv, 0.045), frameMat);
    b.position.copy(toWorld(u, v, depth));
    b.quaternion.copy(q);
    g.add(b);
  };
  const h = o.spring - o.v0;
  addBar(o.u0 + bar / 2, o.v0 + h / 2, bar, h); // stiles
  addBar(o.u1 - bar / 2, o.v0 + h / 2, bar, h);
  addBar(cu, o.v0 + bar / 2, o.u1 - o.u0, bar); // bottom rail
  addBar(cu, o.spring - bar / 2, o.u1 - o.u0, bar * 1.4); // transom bar at springline
  addBar(cu, 2.0, o.u1 - o.u0, bar * 1.6); // meeting rail between sash and transom lights
  addBar(cu, o.v0 + (2.0 - o.v0) / 2, bar * 0.8, 2.0 - o.v0); // center mullion
  addBar(cu, 2.0 + (o.spring - 2.0) / 2, bar * 0.6, o.spring - 2.0); // transom mullion
  // arched head
  const archPath = arc(cu, o.spring, r - bar / 2, Math.PI, 0, 24).map(([u, v]) => toWorld(u, v, depth));
  const archProfile: [number, number][] = [[-bar / 2, -0.022], [bar / 2, -0.022], [bar / 2, 0.022], [-bar / 2, 0.022]];
  g.add(mesh(sweep(archProfile, archPath, w.n, { closed: false }), frameMat));

  // glass
  const glassShape = new THREE.Shape();
  glassShape.moveTo(o.u0, o.v0);
  glassShape.lineTo(o.u0, o.spring);
  glassShape.absarc(cu, o.spring, r, Math.PI, 0, true);
  glassShape.lineTo(o.u1, o.v0);
  glassShape.closePath();
  const glass = mesh(new THREE.ShapeGeometry(glassShape, 24), m.glass);
  glass.castShadow = false;
  glass.applyMatrix4(wallMatrix(w));
  glass.position.addScaledVector(w.n, depth - 0.005);
  g.add(glass);
  return g;
}
