import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { box, boxUV, cylinder, lathe, mesh, place, type V3 } from '../../packages/room-runtime/helpers';
import { mulberry32 } from '../../packages/room-runtime/materials';
import type { TroneMaterials } from './materials';

type Physics = { type: 'static' } | { type: 'dynamic'; mass: number; shape?: 'box' | 'cylinder' | 'hull' | 'sphere' };
const tag = <T extends THREE.Object3D>(o: T, physics: Physics, name: string) => {
  o.userData.physics = physics;
  o.name = name;
  return o;
};

/** A flat sheet leaning on a wall: bottom edge `out` metres from the wall, top resting on it. Local +z faces the room. */
function leaning(w: number, h: number, t: number, out: number, material: THREE.Material) {
  const g = new THREE.Group();
  g.userData.exportName = 'leaning';
  const tilt = Math.asin(Math.min(0.9, out / h));
  const sheet = mesh(boxUV(new THREE.BoxGeometry(w, h, t)), material);
  sheet.position.set(0, (h / 2) * Math.cos(tilt), out - (h / 2) * Math.sin(tilt) + t / 2);
  sheet.rotation.x = -tilt;
  g.add(sheet);
  return g;
}

/** Floor lamp: weighted disc, thin black pole, optional drum shade (photos 180, 186, 209, 287). */
function floorLamp(m: TroneMaterials, shade: boolean) {
  const g = new THREE.Group();
  g.userData.exportName = 'floor-lamp';
  g.add(cylinder(0.14, 0.025, [0, 0.0125, 0], m.lampMetal, 32));
  const top = shade ? 1.5 : 1.52;
  g.add(cylinder(0.009, top, [0, top / 2, 0], m.lampMetal, 10));
  if (shade) {
    g.add(cylinder(0.012, 0.25, [0, 1.25, 0], m.brass, 10));
    const drum = mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.25, 40, 1, true), m.shade, 'shade');
    drum.position.y = 1.62;
    drum.castShadow = false;
    g.add(drum);
    g.add(place(mesh(new THREE.SphereGeometry(0.04, 12, 8), m.bulb), [0, 1.6, 0]));
  } else {
    g.add(cylinder(0.02, 0.06, [0, top + 0.03, 0], m.lampMetal, 12));
    // chrome ball reflector, open towards the room, with the bulb glowing in it (photo 290)
    const dome = mesh(new THREE.SphereGeometry(0.085, 20, 14, 0, Math.PI * 2, 0, Math.PI * 0.7), m.chrome, 'dome');
    dome.position.y = top + 0.1;
    dome.rotation.x = -Math.PI / 2;
    const bulb = mesh(new THREE.SphereGeometry(0.04, 16, 12), m.bulb, 'bulb');
    bulb.position.set(0, top + 0.1, 0.03);
    bulb.castShadow = dome.castShadow = false;
    g.add(dome, bulb);
  }
  return g;
}

/** Grey storage crate with offcuts sticking out (photos 209, 210, 288). */
function crate(m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'crate';
  const w = 0.6, d = 0.4, h = 0.32, t = 0.012;
  g.add(box([w, t, d], [0, t / 2, 0], m.greyPlastic));
  for (const [sx, sz, bw, bd] of [[0, d / 2, w, t], [0, -d / 2, w, t], [w / 2, 0, t, d], [-w / 2, 0, t, d]] as const) {
    g.add(box([bw + 0.01, h, bd + 0.01], [sx, h / 2, sz], m.greyPlastic));
  }
  const rnd = mulberry32(21);
  for (let i = 0; i < 14; i++) {
    const len = 0.35 + rnd() * 0.5;
    const s = box([0.04 + rnd() * 0.08, len, 0.018], [(rnd() - 0.5) * 0.45, 0.1 + len / 2, (rnd() - 0.5) * 0.28], rnd() < 0.5 ? m.pine : m.plywood);
    s.rotation.set((rnd() - 0.5) * 0.7, rnd() * Math.PI, (rnd() - 0.5) * 0.8);
    g.add(s);
  }
  // the white sign board lying on top (photo 288)
  const sign = box([0.22, 0.3, 0.01], [0.05, 0.5, 0], m.paper);
  sign.rotation.set(-0.5, 0.3, 0.2);
  g.add(sign);
  return g;
}

/** Quarter-round corner shower tray: square corner at the origin, curved front toward +x/+z (photo 102). */
function showerTray(m: TroneMaterials) {
  const s = new THREE.Shape();
  s.moveTo(0, 0); s.lineTo(0.8, 0); s.absarc(0, 0, 0.8, 0, Math.PI / 2, false); s.lineTo(0, 0);
  const geo = new THREE.ExtrudeGeometry(s, { depth: 0.05, bevelEnabled: true, bevelSize: 0.01, bevelThickness: 0.01, bevelSegments: 2, curveSegments: 24 });
  geo.rotateX(Math.PI / 2);
  geo.translate(0, 0.06, 0);
  return mesh(geo, m.whitePlastic, 'shower-tray');
}

/** Woven polypropylene rubble sack, slumped (photos 107, 112, 113). */
function sack(m: TroneMaterials, seed: number) {
  const rnd = mulberry32(seed);
  const geo = new RoundedBoxGeometry(0.4, 0.74, 0.3, 3, 0.1);
  const pos = geo.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const bulge = 1 + 0.1 * Math.cos((y / 0.74) * Math.PI) + (rnd() - 0.5) * 0.05;
    pos.setX(i, pos.getX(i) * bulge); pos.setZ(i, pos.getZ(i) * bulge);
    if (y > 0.2) pos.setX(i, pos.getX(i) * (1 - (y - 0.2) * 0.9));
  }
  geo.translate(0, 0.37, 0);
  geo.computeVertexNormals();
  boxUV(geo);
  return mesh(geo, m.sack, 'sack');
}

/** Paint bucket: tapered tub with a lid rim. */
function bucket(material: THREE.Material, r = 0.15, h = 0.33) {
  const g = new THREE.Group();
  g.name = 'bucket';
  g.add(lathe([[0, 0], [r * 0.86, 0], [r, h], [r * 1.04, h], [r * 1.04, h + 0.02], [0, h + 0.02]], material, 28));
  return g;
}

/** Folding hand trolley (photos 107, 108, 112). */
function trolley(m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'trolley';
  const tilt = 0.18;
  const frame = new THREE.Group();
  for (const x of [-0.17, 0.17]) frame.add(box([0.022, 1.0, 0.022], [x, 0.5, 0], m.chrome));
  for (const y of [0.25, 0.6, 0.98]) frame.add(box([0.36, 0.018, 0.018], [0, y, 0], m.chrome));
  frame.add(box([0.38, 0.03, 0.03], [0, 1.0, 0], m.blackPlastic));
  frame.rotation.x = -tilt;
  g.add(frame);
  g.add(box([0.36, 0.01, 0.2], [0, 0.03, 0.1], m.chrome));
  for (const x of [-0.2, 0.2]) {
    const w = mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.05, 20), m.blackPlastic);
    w.rotation.z = Math.PI / 2;
    w.position.set(x, 0.08, -0.08);
    g.add(w);
  }
  return g;
}

/** Cable reel on its stand (photos 166, 167). */
function cableReel(m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'cable-reel';
  const drum = mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.2, 24), m.cableOrange);
  drum.rotation.z = Math.PI / 2;
  drum.position.y = 0.2;
  g.add(drum);
  for (const x of [-0.11, 0.11]) {
    const disc = mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.015, 28), m.blackPlastic);
    disc.rotation.z = Math.PI / 2;
    disc.position.set(x, 0.2, 0);
    g.add(disc);
  }
  g.add(box([0.3, 0.02, 0.3], [0, 0.01, 0], m.blackPlastic));
  return g;
}

/** Bundle of black steel rods leaning on a wall (photos 141, 180, 186). */
function rods(m: TroneMaterials, n: number, seed: number) {
  const g = new THREE.Group();
  g.name = 'rods';
  const rnd = mulberry32(seed);
  for (let i = 0; i < n; i++) {
    const h = 1.6 + rnd() * 0.5, r = 0.012;
    const rod = mesh(new THREE.CylinderGeometry(r, r, h, 8), m.blackPipe);
    rod.position.set((rnd() - 0.5) * 0.25, h / 2 * Math.cos(0.15), 0.15 - (h / 2) * Math.sin(0.15));
    rod.rotation.set(-0.15, 0, (rnd() - 0.5) * 0.15);
    g.add(rod);
  }
  return g;
}

/** Place `obj` against a straight wall stretch: local +z faces the room along `n` (unit x/z). */
function against(obj: THREE.Object3D, x: number, z: number, n: [number, number]) {
  obj.position.set(x, 0, z);
  obj.rotation.y = Math.atan2(n[0], n[1]);
  return obj;
}

export function buildClutter(m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'clutter';
  const N: [number, number] = [0, 1], S: [number, number] = [0, -1];

  // north wall, x -0.5..2.6: boards stood against the wall, crate and bare-bulb lamp (photos 209-212, 287)
  // board feet ray-cast from photo 212
  const boards: [number, number, number, number, THREE.Material][] = [
    [-0.45, 0.6, 1.0, 0.2, m.chipboard], [0.15, 0.62, 1.1, 0.22, m.chipboard], [0.95, 0.5, 1.62, 0.2, m.paperboard],
    [1.4, 0.45, 1.5, 0.25, m.chipboard], [1.75, 0.35, 1.1, 0.3, m.paperboard],
  ];
  for (const [x, w, h, out, mat] of boards) g.add(tag(against(leaning(w, h, 0.018, out, mat), x, -3.0, N), { type: 'static' }, 'board'));
  g.add(tag(against(leaning(0.4, 0.4, 0.02, 0.08, m.steel), -1.3, -2.95, N), { type: 'dynamic', mass: 1, shape: 'box' }, 'loose-grille'));
  g.add(tag(against(crate(m), 2.3, -2.72, N), { type: 'dynamic', mass: 6, shape: 'box' }, 'crate'));
  g.add(tag(place(floorLamp(m, false), [2.64, 0, -1.76]), { type: 'dynamic', mass: 4, shape: 'hull' }, 'lamp-bulb'));
  // timber stair stringers stacked against the wall (photo 288)
  for (const [x, rot] of [[3.0, 0.1], [3.25, -0.05]] as const) {
    const s = leaning(0.35, 0.9, 0.04, 0.3, m.pine);
    s.rotation.z = rot;
    g.add(tag(against(s, x, -2.96, N), { type: 'static' }, 'stringer'));
  }

  // north-east bay: drum-shade lamp, black board and rods (photos 180, 186, 188)
  g.add(tag(place(floorLamp(m, true), [5.94, 0, -1.3]), { type: 'dynamic', mass: 5, shape: 'hull' }, 'lamp-drum'));
  g.add(tag(against(leaning(0.45, 0.42, 0.04, 0.08, m.blackPlastic), 6.05, -2.77, N), { type: 'static' }, 'black-board'));
  g.add(tag(against(rods(m, 7, 3), 6.65, -2.77, N), { type: 'static' }, 'rods'));
  g.add(tag(against(rods(m, 5, 8), 8.5, -2.6, [-1, 0]), { type: 'static' }, 'rods-east'));

  // south wall: black faux-brick panels and planks (photos 76, 79, 99)
  for (const [x, out] of [[1.45, 0.25], [2.05, 0.3], [2.6, 0.22]] as const) {
    g.add(tag(against(leaning(0.62, 1.15, 0.03, out, m.fauxBrick), x, 2.93, S), { type: 'static' }, 'faux-brick'));
  }
  for (const [x, z, l, r] of [[0.95, 2.6, 0.9, 0.2], [1.1, 2.5, 0.7, -0.3], [1.3, 2.62, 0.6, 1.4]] as const) {
    g.add(tag(place(box([l, 0.06, 0.12], [0, 0.03, 0], m.pine), [x, 0, z], r), { type: 'dynamic', mass: 2, shape: 'box' }, 'plank'));
  }
  // paint tins, shower tray and plasterboard sheet (photos 102, 200)
  g.add(tag(against(leaning(1.2, 1.25, 0.013, 0.2, m.paperboard), 4.3, 2.8, S), { type: 'static' }, 'plasterboard-sheet'));
  g.add(tag(place(showerTray(m), [4.75, 0, 2.72], Math.PI), { type: 'dynamic', mass: 8, shape: 'hull' }, 'shower-tray'));
  for (const [x, y, z, mat, r, h] of [[5.15, 0, 2.45, m.whitePlastic, 0.15, 0.33], [4.5, 0.06, 2.5, m.chrome, 0.11, 0.2], [4.3, 0.06, 2.4, m.whitePlastic, 0.14, 0.3], [4.95, 0, 2.25, m.chrome, 0.12, 0.22]] as const) {
    g.add(tag(place(bucket(mat, r, h), [x, y, z]), { type: 'dynamic', mass: 3, shape: 'cylinder' }, 'bucket'));
  }
  g.add(tag(place(new THREE.Group().add(box([0.55, 0.42, 0.18], [0, 0.21, 0], m.blackPlastic)), [3.7, 0, 2.5], 0.1), { type: 'dynamic', mass: 3, shape: 'box' }, 'black-bag'));

  // south-east: rubble sacks, trolley and the cable reel by the column (photos 107-113, 166)
  // sack feet ray-cast from photo 113
  // sack feet ray-cast from photos 112 and 113: three in a row, a bag of rubble and a black bag behind
  const sacks: V3[] = [[6.37, 2.25, 1.6], [6.34, 1.95, 1.5], [6.31, 1.67, 1.7]];
  sacks.forEach(([x, z, r], i) => g.add(tag(place(sack(m, 30 + i), [x, 0, z], r), { type: 'dynamic', mass: 20, shape: 'hull' }, 'sack')));
  const rubble = mesh(new THREE.SphereGeometry(0.22, 16, 12), m.blackPlastic, 'rubble-bag');
  rubble.scale.set(1, 0.8, 1); rubble.position.y = 0.17;
  g.add(tag(place(new THREE.Group().add(rubble), [6.36, 0, 1.3]), { type: 'dynamic', mass: 15, shape: 'sphere' }, 'rubble-bag'));
  g.add(tag(place(new THREE.Group().add(box([0.5, 0.75, 0.2], [0, 0.375, 0], m.blackPlastic)), [6.12, 0, 1.95], Math.PI / 2 - 0.2), { type: 'dynamic', mass: 4, shape: 'box' }, 'black-bag'));
  g.add(tag(place(trolley(m), [6.72, 0, 2.5], -Math.PI / 2 - 0.4), { type: 'dynamic', mass: 6, shape: 'box' }, 'trolley'));
  g.add(tag(place(cableReel(m), [7.84, 0, 0.95], 1.2), { type: 'dynamic', mass: 5, shape: 'box' }, 'cable-reel'));
  return g;
}
