import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { box, boxUV, cushion, mesh } from '../../packages/room-runtime/helpers';
import { mulberry32 } from '../../packages/room-runtime/materials';
import { posterTexture, type AtriumMaterials } from './materials';

type V3 = [number, number, number];
const UP = new THREE.Vector3(0, 1, 0);

/** Straight tapered rod between two points. */
function rod(a: V3, b: V3, rTop: number, rBottom: number, material: THREE.Material) {
  const from = new THREE.Vector3(...a), to = new THREE.Vector3(...b);
  const dir = to.clone().sub(from);
  const m = mesh(new THREE.CylinderGeometry(rTop, rBottom, dir.length(), 10), material);
  m.position.copy(from).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(UP, dir.normalize());
  return m;
}

/**
 * Shell sofa/armchair of the "About A Lounge" family (photos 44, 68, 124, 136): a horseshoe shell
 * whose arms fall toward the front, a seat cushion, and four splayed oak legs. Front is +z.
 */
export function loungeSeat(o: { w: number; d: number; back: number; shell: THREE.Material }, m: AtriumMaterials) {
  const { w, d } = o;
  const g = new THREE.Group();
  g.userData.exportName = 'lounge-seat';
  const baseY = 0.22, arm = 0.13, rb = Math.min(0.32, w * 0.25), ri = Math.max(0.05, rb - arm);
  const s = new THREE.Shape();
  s.moveTo(-w / 2, -d / 2);
  s.lineTo(-w / 2, d / 2 - rb); s.quadraticCurveTo(-w / 2, d / 2, -w / 2 + rb, d / 2);
  s.lineTo(w / 2 - rb, d / 2); s.quadraticCurveTo(w / 2, d / 2, w / 2, d / 2 - rb);
  s.lineTo(w / 2, -d / 2); s.lineTo(w / 2 - arm, -d / 2);
  s.lineTo(w / 2 - arm, d / 2 - arm - ri); s.quadraticCurveTo(w / 2 - arm, d / 2 - arm, w / 2 - arm - ri, d / 2 - arm);
  s.lineTo(-w / 2 + arm + ri, d / 2 - arm); s.quadraticCurveTo(-w / 2 + arm, d / 2 - arm, -w / 2 + arm, d / 2 - arm - ri);
  s.lineTo(-w / 2 + arm, -d / 2); s.closePath();
  const height = o.back - baseY - 0.04;
  let geo: THREE.BufferGeometry = new THREE.ExtrudeGeometry(s, {
    depth: height, bevelEnabled: true, bevelThickness: 0.07, bevelSize: 0.07, bevelOffset: -0.07, bevelSegments: 5, curveSegments: 12,
  });
  geo.rotateX(-Math.PI / 2);
  const pos = geo.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    // arms fall toward the front: scale height by distance from the back
    const t = (d / 2 - pos.getZ(i)) / d;
    pos.setY(i, pos.getY(i) * (0.8 + 0.2 * Math.min(1, Math.max(0, t))) + baseY);
  }
  // tub-like bulge: the shell swells at mid height
  const bulge = Math.PI / (o.back - baseY);
  for (let i = 0; i < pos.count; i++) {
    const k = 1 + 0.07 * Math.sin(Math.min(Math.PI, Math.max(0, (pos.getY(i) - baseY) * bulge)));
    pos.setX(i, pos.getX(i) * k); pos.setZ(i, pos.getZ(i) * k);
  }
  geo.deleteAttribute('normal'); geo.deleteAttribute('uv');
  geo = mergeVertices(geo, 1e-3);
  geo.computeVertexNormals();
  boxUV(geo);
  g.add(mesh(geo, o.shell, 'shell'));
  g.add(cushion([w - 0.04, 0.13, d - 0.05], [0, baseY + 0.07, 0], 0.05, o.shell));
  g.add(cushion([w - 0.2, 0.1, d - 0.16], [0, 0.37, 0.04], 0.045, o.shell));
  const lx = w / 2 - 0.12, lz = d / 2 - 0.12;
  for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    g.add(rod([sx * lx, baseY + 0.02, sz * lz], [sx * (lx + 0.07), 0, sz * (lz + 0.07)], 0.024, 0.014, m.oakLeg));
  }
  return g;
}

/** Retro two-seater: button-tufted back, rolled arms, slim flared legs (photos 15, 16, 20). */
export function tuftedSofa(w: number, d: number, m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'tufted-sofa';
  g.add(cushion([w, 0.26, d], [0, 0.31, 0], 0.07, m.yellowFabric));
  g.add(cushion([w - 0.36, 0.13, d - 0.2], [0, 0.5, 0.07], 0.05, m.yellowFabric));
  const back = cushion([w - 0.12, 0.4, 0.2], [0, 0.68, -d / 2 + 0.1], 0.07, m.tuftedYellow);
  back.rotation.x = -0.14;
  g.add(back);
  for (const sx of [-1, 1]) g.add(cushion([0.17, 0.34, d - 0.02], [sx * (w / 2 - 0.085), 0.58, 0], 0.07, m.yellowFabric));
  for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    g.add(rod([sx * (w / 2 - 0.1), 0.2, sz * (d / 2 - 0.1)], [sx * (w / 2 - 0.04), 0, sz * (d / 2 - 0.04)], 0.02, 0.012, m.oakLeg));
  }
  return g;
}

/** Low table with a pale top and three splayed oak legs (photos 44, 68, 20). */
export function roundTable(d: number, h: number, m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'round-table';
  const top = mesh(new THREE.CylinderGeometry(d / 2, d / 2 - 0.012, 0.035, 40), m.tableTop);
  top.position.y = h - 0.0175;
  g.add(top);
  for (let i = 0; i < 3; i++) {
    const a = (i / 3) * Math.PI * 2 + 0.5;
    g.add(rod([Math.cos(a) * d * 0.16, h - 0.03, Math.sin(a) * d * 0.16], [Math.cos(a) * d * 0.4, 0, Math.sin(a) * d * 0.4], 0.02, 0.012, m.oakLeg));
  }
  return g;
}

/** Oak-top coffee table on a black rod frame with a lower shelf rail (photo 124). */
export function frameTable(w: number, d: number, h: number, m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'frame-table';
  const t = 0.04, bar = 0.026;
  g.add(box([w, t, d], [0, h - t / 2, 0], m.oakTop));
  const hx = w / 2 - bar / 2 - 0.02, hz = d / 2 - bar / 2 - 0.02;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) g.add(box([bar, h - t, bar], [sx * hx, (h - t) / 2, sz * hz], m.steel));
  for (const y of [h - t - bar / 2, 0.14]) {
    for (const sz of [-1, 1]) g.add(box([2 * hx, bar, bar], [0, y, sz * hz], m.steel));
    for (const sx of [-1, 1]) g.add(box([bar, bar, 2 * hz], [sx * hx, y, 0], m.steel));
  }
  return g;
}

/** Upholstered cubes with chrome feet: one instanced mesh per part, tinted per instance (photos 44, 68, 124). */
export function poufs(list: [number, number, number, string][], m: AtriumMaterials) {
  const g = new THREE.Group();
  g.name = 'poufs';
  const body = new THREE.InstancedMesh(new RoundedBoxGeometry(0.44, 0.38, 0.44, 4, 0.075), m.poufFabric, list.length);
  const feet = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.022, 0.026, 0.03, 10), m.chrome, list.length * 4);
  const mat = new THREE.Matrix4(), tmp = new THREE.Matrix4(), color = new THREE.Color();
  list.forEach(([x, z, rot, hex], i) => {
    mat.makeRotationY(rot).setPosition(x, 0.04 + 0.19, z);
    body.setMatrixAt(i, mat);
    body.setColorAt(i, color.set(hex));
    [[1, 1], [-1, 1], [1, -1], [-1, -1]].forEach(([sx, sz], k) => {
      tmp.makeRotationY(rot).setPosition(x, 0, z).multiply(new THREE.Matrix4().makeTranslation(sx * 0.17, 0.015, sz * 0.17));
      feet.setMatrixAt(i * 4 + k, tmp);
    });
  });
  for (const im of [body, feet]) { im.castShadow = im.receiveShadow = true; im.computeBoundingSphere(); }
  g.add(body, feet);
  return g;
}

/** Black drum-shade floor lamp with a brass collar (photos 4, 44, 72). */
export function floorLamp(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'floor-lamp';
  g.add(mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.02, 28), m.steel));
  g.add(mesh(new THREE.CylinderGeometry(0.008, 0.008, 1.38, 8), m.steel).translateY(0.7));
  g.add(mesh(new THREE.CylinderGeometry(0.013, 0.013, 0.07, 10), m.brass).translateY(1.25));
  const shade = mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.29, 36, 1, true), m.lampShade);
  shade.position.y = 1.55;
  shade.castShadow = false;
  g.add(shade);
  return g;
}

/** Hall tree with a trench coat, a dark coat and a hat (photos 24, 52). */
export function coatStand(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'coat-stand';
  g.add(mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.02, 24), m.steel));
  g.add(mesh(new THREE.CylinderGeometry(0.014, 0.014, 1.72, 8), m.steel).translateY(0.87));
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + 0.4;
    g.add(rod([0, 1.7, 0], [Math.cos(a) * 0.14, 1.82, Math.sin(a) * 0.14], 0.01, 0.01, m.steel));
  }
  const coat = (color: string, x: number, z: number, h: number, w: number) => {
    const c = cushion([w, h, 0.1], [x, 1.62 - h / 2, z], 0.04, m.solid(color, 0.95));
    c.rotation.y = 0.5;
    return c;
  };
  g.add(coat('#b7a07a', 0.1, 0.06, 0.95, 0.32), coat('#2a292b', -0.1, -0.07, 1.15, 0.3));
  const hat = mesh(new THREE.SphereGeometry(0.1, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2), m.solid('#8c1f2a', 0.9));
  hat.position.set(0.1, 1.74, 0.12);
  g.add(hat);
  return g;
}

/** Four-tier dark-wood ladder bookcase with X-braced sides, game boxes and an autumn garland (photo 169). */
export function bookcase(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'bookcase';
  const W = 0.86, D = 0.34, H = 1.68, p = 0.04;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) g.add(box([p, H, p], [sx * (W / 2 - p / 2), H / 2, sz * (D / 2 - p / 2)], m.walnut));
  const tiers = [0.12, 0.6, 1.0, 1.4, H - 0.03];
  for (const y of tiers) g.add(box([W, 0.035, D], [0, y, 0], m.walnut));
  for (const sx of [-1, 1]) {
    for (const k of [1, -1]) {
      g.add(rod([sx * (W / 2 - p / 2), 0.16, -k * (D / 2 - p / 2)], [sx * (W / 2 - p / 2), H - 0.1, k * (D / 2 - p / 2)], 0.013, 0.013, m.walnut));
    }
  }
  const rnd = mulberry32(7);
  const colors = ['#7d3a32', '#b89a58', '#232326', '#3e5a68', '#5d4565', '#a8592b', '#cfc7b5', '#4f5e44'];
  for (const y of tiers.slice(0, 4)) {
    let x = -W / 2 + 0.07;
    while (x < W / 2 - 0.12) {
      const w = 0.04 + rnd() * 0.1, h = 0.13 + rnd() * 0.16;
      if (rnd() < 0.3) { x += w + 0.06; continue; }
      if (x + w > W / 2 - 0.05) break;
      g.add(box([w, h, 0.12 + rnd() * 0.1], [x + w / 2, y + 0.0175 + h / 2, -0.02], m.solid(colors[Math.floor(rnd() * colors.length)], 0.8)));
      x += w + 0.01 + rnd() * 0.03;
    }
  }
  garland(g, [-W / 2 + 0.05, H, -0.08], W - 0.1, m, 11);
  return g;
}

/** Scatter of orange leaves along a run (the autumn garlands on the shelves and ducts). */
function garland(g: THREE.Group, from: V3, length: number, m: AtriumMaterials, seed: number) {
  const rnd = mulberry32(seed);
  const leaves = new THREE.InstancedMesh(new THREE.CircleGeometry(0.035, 5), m.leaf, 16);
  const mat = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
  for (let i = 0; i < 16; i++) {
    q.setFromEuler(new THREE.Euler(-Math.PI / 2 + (rnd() - 0.5) * 0.6, rnd() * 6, (rnd() - 0.5) * 0.6));
    mat.compose(new THREE.Vector3(from[0] + (i / 15) * length, from[1] + 0.015 + rnd() * 0.04, from[2] + (rnd() - 0.5) * 0.12), q, new THREE.Vector3(1 + rnd(), 1 + rnd(), 1));
    leaves.setMatrixAt(i, mat);
    leaves.setColorAt(i, c.setHSL(0.05 + rnd() * 0.07, 0.8, 0.45));
  }
  g.add(leaves);
}

/** White 4 x 3 cube unit with coloured doors (photos 34, 36, 84). */
export function cubeShelf(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'cube-shelf';
  const W = 1.9, H = 1.45, D = 0.36, t = 0.022, cols = 4, rows = 3;
  const cw = (W - t * (cols + 1)) / cols, ch = (H - t * (rows + 1)) / rows;
  g.add(box([W, t, D], [0, H - t / 2, 0], m.white), box([W, t, D], [0, t / 2, 0], m.white));
  for (let c = 0; c <= cols; c++) g.add(box([t, H, D], [-W / 2 + t / 2 + c * (cw + t), H / 2, 0], m.white));
  for (let r = 1; r < rows; r++) g.add(box([W, t, D], [0, t + r * (ch + t) - t / 2, 0], m.white));
  g.add(box([W, H, 0.01], [0, H / 2, -D / 2 + 0.005], m.white));
  const red = m.solid('#d5261b', 0.45), oak = m.solid('#cdb48c', 0.6), white = m.white, dark = m.solid('#1a1a1a', 0.8);
  const pattern = [[white, oak, white, oak], [red, white, red, white], [oak, red, oak, red]];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = -W / 2 + t + cw / 2 + c * (cw + t), y = H - t - ch / 2 - r * (ch + t);
      g.add(box([cw - 0.006, ch - 0.006, 0.018], [x, y, D / 2 - 0.009], pattern[r][c]));
      g.add(box([0.11, 0.012, 0.004], [x, y + ch / 2 - 0.045, D / 2 + 0.002], dark));
    }
  }
  garland(g, [-W / 2 + 0.1, H, 0], W - 0.2, m, 3);
  return g;
}

/** Framed print on the wall: local +z is out of the wall. */
export function framed(w: number, h: number, palette: string[], seed: number, m: AtriumMaterials, frame = true) {
  const g = new THREE.Group();
  g.userData.exportName = 'framed';
  const print = mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: posterTexture(palette, seed), roughness: 0.7 }));
  print.castShadow = false;
  print.position.z = frame ? 0.016 : 0.003;
  g.add(print);
  if (frame) {
    const f = 0.022;
    g.add(box([w + 2 * f, f, 0.03], [0, h / 2 + f / 2, 0.015], m.steel), box([w + 2 * f, f, 0.03], [0, -h / 2 - f / 2, 0.015], m.steel));
    g.add(box([f, h, 0.03], [-w / 2 - f / 2, 0, 0.015], m.steel), box([f, h, 0.03], [w / 2 + f / 2, 0, 0.015], m.steel));
  }
  return g;
}

/** Wall-hung panel radiator (photos 4, 96, 100). */
/** Red dry-powder extinguisher on the wall (photo 0). */
export function extinguisher(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'extinguisher';
  g.add(mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.4, 16), m.solid('#c4161c', 0.4)).translateY(0.2).translateZ(0.08));
  g.add(mesh(new THREE.CylinderGeometry(0.03, 0.04, 0.09, 12), m.steel).translateY(0.44).translateZ(0.08));
  return g;
}

export function radiator(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'radiator';
  g.add(box([0.85, 0.5, 0.07], [0, 0, 0.035], m.radiator));
  g.add(box([0.78, 0.38, 0.01], [0, 0, 0.075], m.solid('#a9aeb1', 0.6)));
  return g;
}

export function laptop(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'laptop';
  g.add(box([0.32, 0.012, 0.22], [0, 0.006, 0], m.solid('#7d8186', 0.4, 0.7)));
  const lid = box([0.32, 0.22, 0.008], [0, 0.12, -0.11], m.solid('#2c2e31', 0.4, 0.5));
  lid.rotation.x = -0.35;
  g.add(lid);
  return g;
}

export function basket(m: AtriumMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'basket';
  g.add(mesh(new THREE.CylinderGeometry(0.17, 0.15, 0.36, 24), m.solid('#b9955b', 0.95)).translateY(0.18));
  g.add(mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.02, 24), m.solid('#e9e5d8', 0.7)).translateY(0.37));
  return g;
}

export function backpack(m: AtriumMaterials) {
  return cushion([0.32, 0.12, 0.42], [0, 0.07, 0], 0.05, m.solid('#1c1d20', 0.9));
}

