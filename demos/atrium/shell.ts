import * as THREE from 'three';
import { buildWall, onWall, wallMatrix, wallPoint, type WallDef } from '../../packages/room-runtime/walls';
import { box, boxUV, mesh } from '../../packages/room-runtime/helpers';
import { pointInPolygon, polygonEdges, polygonSurface } from '../../packages/room-runtime/polygon-room';
import type { AtriumMaterials } from './materials';

export const HEIGHT = 2.39;
const THICKNESS = 0.2;

/**
 * Floor outline measured on the LiDAR mesh (vertical-surface density, 1 cm cells). The RoomPlan walls
 * were off by 0.15 m (north), 0.1-0.55 m (south, which also bends at x = -5.65 and 3.4) and the
 * 0.28 m "wall 1" duplicates the east wall, so they are not used for geometry.
 */
export const POLY: [number, number][] = [
  [5.97, 0.31], [7.88, 0.31], [7.88, -2.55], [-7.86, -2.43], [-7.86, 2.27], [-5.65, 2.87], [3.4, 2.84], [5.97, 2.72],
];

export interface OpeningSpec {
  edge: number;
  /** world x/z of the opening's centre on the wall */
  at: [number, number];
  width: number;
  height: number;
  kind: 'door' | 'passage';
  /** door leaf: closed, or swung open by this angle (rad) around the hinge side */
  open?: { hinge: 'start' | 'end'; angle: number };
  closer?: boolean;
  /** what shows through the opening: a blue-lit room, a lit utility room, or a dark one (default) */
  beyond?: 'blue' | 'lit';
  steel?: boolean;
}

/** RoomPlan openings and the closed south-wall doors seen in the scan elevation and photos 4, 28, 56. */
export const OPENINGS: OpeningSpec[] = [
  { edge: 1, at: [7.88, -0.41], width: 1.2, height: 2.08, kind: 'door', steel: true, beyond: 'lit', open: { hinge: 'end', angle: -1.5 } },
  { edge: 2, at: [-5.26, -2.46], width: 0.86, height: 2.22, kind: 'door', closer: true },
  { edge: 2, at: [4.61, -2.52], width: 0.92, height: 2.06, kind: 'passage' },
  { edge: 2, at: [7.15, -2.55], width: 1.33, height: 2.08, kind: 'passage', beyond: 'lit' },
  { edge: 3, at: [-7.86, -0.585], width: 1.24, height: 2.0, kind: 'passage' },
  { edge: 5, at: [-5.12, 2.87], width: 0.92, height: 2.1, kind: 'door', closer: true },
  { edge: 5, at: [-3.23, 2.86], width: 0.95, height: 2.1, kind: 'door', closer: true },
  { edge: 5, at: [0.73, 2.87], width: 0.79, height: 2.1, kind: 'door', open: { hinge: 'end', angle: 1.2 }, beyond: 'blue' },
  { edge: 7, at: [5.97, 2.14], width: 0.86, height: 2.11, kind: 'door', open: { hinge: 'end', angle: 1.3 }, beyond: 'blue' },
];

export const insideRoom = (x: number, z: number) => pointInPolygon(POLY, x, z);

export interface ShellWall { def: WallDef; group: THREE.Group; specs: (OpeningSpec & { u0: number; u1: number })[] }

function buildDoor(def: WallDef, o: OpeningSpec & { u0: number; u1: number }, m: AtriumMaterials) {
  const g = new THREE.Group();
  g.name = 'door';
  const w = o.u1 - o.u0 - 0.03, h = o.height - 0.015;
  const leafOut = -0.1;
  const pivot = new THREE.Group();
  const leaf = boxUV(new THREE.BoxGeometry(w, h, 0.042));
  const leafMesh = mesh(leaf, o.steel ? m.doorSteel : m.door);
  const startHinge = (o.open?.hinge ?? 'end') === 'start';
  leafMesh.position.set(startHinge ? w / 2 : -w / 2, h / 2, 0);
  pivot.add(leafMesh);
  // lever handle on the free edge, plus the closer arm on the hinge side
  const free = startHinge ? w : -w;
  const handle = mesh(new THREE.BoxGeometry(0.12, 0.022, 0.026), m.chrome);
  handle.position.set(free + (startHinge ? -0.1 : 0.1), 1.0, 0.045);
  const rose = mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.01, 16), m.chrome);
  rose.rotation.x = Math.PI / 2;
  rose.position.set(free + (startHinge ? -0.055 : 0.055), 1.0, 0.025);
  pivot.add(handle, rose);
  if (o.closer) {
    const body = mesh(new THREE.BoxGeometry(0.28, 0.07, 0.05), m.solid('#8c8c8c', 0.4, 0.6));
    body.position.set(startHinge ? 0.2 : -0.2, h - 0.04, 0.045);
    pivot.add(body);
  }
  pivot.position.set(startHinge ? o.u0 + 0.015 : o.u1 - 0.015, 0, leafOut);
  if (o.open) pivot.rotation.y = startHinge ? -o.open.angle : o.open.angle;
  g.add(pivot);
  // casing on the room side
  const c = 0.05;
  const strips: [number, number, number, number][] = [
    [o.u0 - c / 2, o.height / 2, c, o.height], [o.u1 + c / 2, o.height / 2, c, o.height], [(o.u0 + o.u1) / 2, o.height + c / 2, o.u1 - o.u0 + 2 * c, c],
  ];
  for (const [u, v, sw, sh] of strips) {
    const s = mesh(new THREE.BoxGeometry(sw, sh, 0.012), m.reveal);
    s.position.set(u, v, 0.006);
    g.add(s);
  }
  g.position.copy(wallPoint(def, 0, 0, 0));
  g.quaternion.setFromRotationMatrix(wallMatrix(def));
  return g;
}

/** An unlit room behind a doorway so the opening never shows the viewer background. */
function buildVoid(def: WallDef, o: OpeningSpec & { u0: number; u1: number }, m: AtriumMaterials) {
  const depth = 1.2;
  const b = new THREE.Mesh(new THREE.BoxGeometry(o.u1 - o.u0, o.height, depth), o.beyond === 'blue' ? m.voidBlue : o.beyond === 'lit' ? m.voidLit : m.void);
  b.name = 'beyond';
  onWall(b, def, (o.u0 + o.u1) / 2, o.height / 2, -THICKNESS - depth / 2 + 0.02);
  return b;
}

export function buildShell(m: AtriumMaterials) {
  const walls: ShellWall[] = [];
  for (const [i, { a, dir, normal, length, ext }] of polygonEdges(POLY, THICKNESS).entries()) {
    const specs = OPENINGS.filter(o => o.edge === i).map(o => {
      const c = ((o.at[0] - a[0]) * dir.x + (o.at[1] - a[1]) * dir.z);
      return { ...o, u0: c - o.width / 2, u1: c + o.width / 2 };
    });
    const def: WallDef = {
      name: `atrium-${i}`,
      origin: new THREE.Vector3(a[0], 0, a[1]),
      u: dir, n: normal, length, thickness: THICKNESS, height: HEIGHT,
      openings: specs.map(s => ({ kind: 'rect' as const, u0: s.u0, u1: s.u1, v0: 0, v1: s.height })),
      panels: [], crown: false, ext,
      baseboardGaps: [[0, length]],
      material: m.paint, trimMaterial: m.reveal,
    };
    const group = buildWall(def);
    // flat grey skirting, interrupted by the openings
    let start = 0;
    for (const [g0, g1] of [...specs.map(s => [s.u0 - 0.05, s.u1 + 0.05]).sort((p, q) => p[0] - q[0]), [length, length]]) {
      if (g0 - start > 0.02) {
        const strip = box([g0 - start, 0.075, 0.014], [0, 0, 0], m.baseboard);
        onWall(strip, def, (start + g0) / 2, 0.0375, 0.007);
        group.add(strip);
      }
      start = g1;
    }
    for (const s of specs) {
      if (s.kind === 'door') group.add(buildDoor(def, s, m));
      group.add(buildVoid(def, s, m));
    }
    walls.push({ def, group, specs });
  }
  return {
    walls,
    floor: polygonSurface(POLY, 0, m.floor, 'floor'),
    ceiling: polygonSurface(POLY, HEIGHT, m.ceiling, 'ceiling'),
  };
}
