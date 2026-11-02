import * as THREE from 'three';
import { box, mesh } from '../../packages/room-runtime/helpers';
import type { AtriumMaterials } from './materials';
import { HEIGHT, POLY } from './shell';

/** Tube fixtures triangulated from bright spots in upward photos (lights_from_photos.py), spaced ~2.6 m; the last two continue the row. */
export const TUBE_X = [-6.9, -3.8, -1.05, 1.4, 4.0];
const TRAY_Z = 0.08;
const TRAY_RANGE: [number, number] = [-7.6, 5.4];

const NORTH = { a: POLY[3], b: POLY[2] }; // [-7.86,-2.43] -> [7.88,-2.55]
const WEST_X = -7.86;

function pipe(radius: number, from: THREE.Vector3, to: THREE.Vector3, material: THREE.Material) {
  const dir = to.clone().sub(from);
  const g = new THREE.CylinderGeometry(radius, radius, dir.length(), 20);
  const m = mesh(g, material);
  m.position.copy(from).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return m;
}

export function buildCeilingItems(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.name = 'ceiling-items';
  const zAt = (x: number) => NORTH.a[1] + ((x - NORTH.a[0]) / (NORTH.b[0] - NORTH.a[0])) * (NORTH.b[1] - NORTH.a[1]);

  // spiral duct along the north and west walls (photos 15, 20, 64, 68), with an elbow in the corner
  const R = 0.18, yDuct = HEIGHT - R - 0.03;
  g.add(pipe(R, new THREE.Vector3(-7.86 + 0.24, yDuct, zAt(-7.6) + 0.24), new THREE.Vector3(7.7, yDuct, zAt(7.7) + 0.22), m.duct));
  g.add(pipe(R, new THREE.Vector3(WEST_X + 0.24, yDuct, zAt(-7.6) + 0.24), new THREE.Vector3(WEST_X + 0.24, yDuct, 2.2), m.duct));
  const elbow = mesh(new THREE.SphereGeometry(R, 20, 14), m.duct);
  elbow.position.set(WEST_X + 0.24, yDuct, zAt(-7.6) + 0.24);
  g.add(elbow);
  // clamp rings every 1.8 m
  for (let x = -6.5; x < 7.5; x += 1.8) {
    const ring = mesh(new THREE.CylinderGeometry(R + 0.006, R + 0.006, 0.03, 20), m.clamp);
    ring.rotation.z = Math.PI / 2;
    ring.position.set(x, yDuct, zAt(x) + 0.23);
    g.add(ring);
  }
  g.add(pipe(0.05, new THREE.Vector3(-7.7, HEIGHT - 0.3, zAt(-7.7) + 0.08), new THREE.Vector3(7.8, HEIGHT - 0.3, zAt(7.8) + 0.08), m.duct));
  // thin white conduits above the duct
  for (const [dz, y] of [[0.08, HEIGHT - 0.07], [0.14, HEIGHT - 0.05]] as const) {
    g.add(pipe(0.022, new THREE.Vector3(-7.7, y, zAt(-7.7) + dz), new THREE.Vector3(7.8, y, zAt(7.8) + dz), m.duct));
  }

  // perforated cable tray under the central row of tubes
  const [x0, x1] = TRAY_RANGE;
  const len = x1 - x0, cx = (x0 + x1) / 2, yTray = HEIGHT - 0.16;
  g.add(box([len, 0.012, 0.3], [cx, yTray, TRAY_Z], m.tray));
  g.add(box([len, 0.06, 0.012], [cx, yTray + 0.03, TRAY_Z - 0.15], m.tray));
  g.add(box([len, 0.06, 0.012], [cx, yTray + 0.03, TRAY_Z + 0.15], m.tray));
  for (let x = x0 + 0.4; x < x1; x += 1.5) {
    for (const dz of [-0.15, 0.15]) g.add(pipe(0.006, new THREE.Vector3(x, yTray, TRAY_Z + dz), new THREE.Vector3(x, HEIGHT, TRAY_Z + dz), m.tray));
  }
  for (const x of TUBE_X) {
    const tube = mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.2, 14), m.tube);
    tube.rotation.z = Math.PI / 2;
    tube.position.set(x, yTray - 0.07, TRAY_Z);
    tube.castShadow = false;
    g.add(tube, box([1.24, 0.014, 0.05], [x, yTray - 0.02, TRAY_Z], m.steel));
  }
  return g;
}
