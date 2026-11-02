import * as THREE from 'three';
import { box, mesh, type V3 } from '../../packages/room-runtime/helpers';
import networks from './networks.json';
import type { TroneMaterials } from './materials';
import { HEIGHT } from './shell';

type End = 'wall' | 'slab' | 'floor' | 'box' | 'tee' | 'wye' | 'none';
interface Run {
  id: string; kind: string; material: string; d: number; points: V3[];
  /** elbow centreline radius as a multiple of the diameter */
  bend?: number;
  ends?: [End, End];
  /** spacing (m) of ceiling clamps on near-horizontal segments, of sockets, of tape bands on foam insulation */
  clamps?: number; sockets?: number; sleeves?: number;
  /** heights of wall brackets on a vertical run */
  brackets?: number[];
  coupling?: V3;
}
interface Fitting { type: 'tee' | 'wye' | 'reducer' | 'tie'; at: V3; dir: V3; d: number; material: string }

const UP = new THREE.Vector3(0, 1, 0);
const v = (p: V3) => new THREE.Vector3(...p);
const ceilingAt = (x: number) => (x > 7.5 ? 2.27 : HEIGHT);

/** Short cylinder centred on `c` along `dir`. */
function ring(c: THREE.Vector3, dir: THREE.Vector3, r: number, len: number, material: THREE.Material, radial = 20) {
  const m = mesh(new THREE.CylinderGeometry(r, r, len, radial), material);
  m.position.copy(c);
  m.quaternion.setFromUnitVectors(UP, dir.clone().normalize());
  return m;
}

/** Polyline with every corner replaced by a circular-ish arc (quadratic Bezier) of radius `R`, plus the arc end points. */
function filleted(points: THREE.Vector3[], R: number) {
  const path = new THREE.CurvePath<THREE.Vector3>();
  const joints: { p: THREE.Vector3; dir: THREE.Vector3 }[] = [];
  let start = points[0].clone();
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1], c = points[i], b = points[i + 1];
    const din = c.clone().sub(a).normalize(), dout = b.clone().sub(c).normalize();
    const angle = Math.acos(THREE.MathUtils.clamp(din.dot(dout), -1, 1));
    const t = Math.min(R * Math.tan(angle / 2), a.distanceTo(c) * 0.45, c.distanceTo(b) * 0.45);
    if (angle < 0.02 || t < 1e-3) continue;
    const p0 = c.clone().addScaledVector(din, -t), p1 = c.clone().addScaledVector(dout, t);
    path.add(new THREE.LineCurve3(start, p0));
    path.add(new THREE.QuadraticBezierCurve3(p0, c.clone(), p1));
    joints.push({ p: p0, dir: din }, { p: p1, dir: dout });
    start = p1;
  }
  path.add(new THREE.LineCurve3(start, points[points.length - 1].clone()));
  return { path, joints };
}

function buildRun(run: Run, mat: THREE.Material, m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = run.id;
  const r = run.d / 2;
  const pts = run.points.map(v);
  const { path, joints } = filleted(pts, (run.bend ?? 1) * run.d);
  const length = path.getLength();
  g.add(mesh(new THREE.TubeGeometry(path, Math.max(8, Math.ceil(length / 0.15) + joints.length * 6), r, 18, false), mat, `${run.id}-pipe`));
  // fitting sockets at both ends of every elbow
  for (const j of joints) g.add(ring(j.p, j.dir, r * 1.13, 0.05, mat));
  const segs = pts.slice(1).map((b, i) => ({ a: pts[i], b, dir: b.clone().sub(pts[i]).normalize(), len: pts[i].distanceTo(b) }));
  for (const s of segs) {
    const horizontal = Math.abs(s.dir.y) < 0.3;
    if (run.sockets) for (let t = run.sockets; t < s.len - 0.2; t += run.sockets) g.add(ring(s.a.clone().addScaledVector(s.dir, t), s.dir, r * 1.13, 0.07, mat));
    if (run.sleeves && horizontal) for (let t = run.sleeves / 2; t < s.len - 0.1; t += run.sleeves) g.add(ring(s.a.clone().addScaledVector(s.dir, t), s.dir, r * 1.04, 0.05, m.paper));
    if (run.clamps && horizontal) {
      for (let t = run.clamps / 2; t < s.len; t += run.clamps) {
        const c = s.a.clone().addScaledVector(s.dir, t);
        const clip = mesh(new THREE.TorusGeometry(r + 0.004, 0.006, 6, 24), m.chrome);
        clip.position.copy(c);
        clip.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), s.dir);
        g.add(clip);
        const top = ceilingAt(c.x);
        if (top - c.y - r > 0.01) g.add(ring(new THREE.Vector3(c.x, (top + c.y + r) / 2, c.z), UP, 0.005, top - c.y - r, m.chrome, 6));
      }
    }
  }
  // wall brackets on a vertical run: strut arm back to the wall behind it, plus a clamp ring
  for (const y of run.brackets ?? []) {
    const c = new THREE.Vector3(pts[0].x, y, pts[0].z);
    g.add(ring(c, UP, r + 0.008, 0.03, m.chrome));
    g.add(box([0.04, 0.04, 0.18], [c.x, y, c.z + 0.11], m.chrome));
  }
  // collars where the run passes through a wall, the slab or the floor
  (run.ends ?? []).forEach((e, i) => {
    if (e !== 'wall' && e !== 'slab' && e !== 'floor' && e !== 'box') return;
    const p = i === 0 ? pts[0] : pts[pts.length - 1];
    const q = i === 0 ? pts[1] : pts[pts.length - 2];
    g.add(ring(p, p.clone().sub(q), r * 1.5, 0.012, e === 'box' ? m.skim : m.trim));
  });
  if (run.coupling) {
    const c = v(run.coupling);
    const s = segs.reduce((best, sg) => (distToSeg(c, sg.a, sg.b) < distToSeg(c, best.a, best.b) ? sg : best));
    g.add(ring(c, s.dir, r * 1.16, 0.09, m.brass));
  }
  return g;
}

function distToSeg(p: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3) {
  const ab = b.clone().sub(a);
  const t = THREE.MathUtils.clamp(p.clone().sub(a).dot(ab) / ab.lengthSq(), 0, 1);
  return p.distanceTo(a.clone().addScaledVector(ab, t));
}

/** Tee/Y bodies are a socketed sleeve along the main run; a reducer is a short cone; a tie is a wire twist. */
function buildFitting(f: Fitting, mat: THREE.Material, m: TroneMaterials) {
  const c = v(f.at), dir = v(f.dir).normalize(), r = f.d / 2;
  if (f.type === 'tie') return ring(c, dir, r + 0.003, 0.008, m.chrome, 12);
  if (f.type === 'reducer') {
    const cone = mesh(new THREE.CylinderGeometry(0.04, r * 1.12, 0.12, 20), mat, 'reducer');
    cone.position.copy(c).addScaledVector(dir, -0.06);
    cone.quaternion.setFromUnitVectors(UP, dir);
    return cone;
  }
  const g = new THREE.Group();
  g.name = f.type;
  const len = f.type === 'wye' ? 2.8 * f.d : 2.2 * f.d;
  g.add(ring(c, dir, r * 1.14, len, mat));
  for (const s of [-1, 1]) g.add(ring(c.clone().addScaledVector(dir, (s * len) / 2), dir, r * 1.2, 0.03, mat));
  return g;
}

/** Strut channel (41 mm) across a pipe lane, hung from the slab on two threaded rods. */
function buildRail(from: V3, to: V3, m: TroneMaterials) {
  const g = new THREE.Group();
  g.userData.exportName = 'build-rail';
  const a = v(from), b = v(to), dir = b.clone().sub(a), len = dir.length();
  const c = a.clone().lerp(b, 0.5);
  const rail = mesh(new THREE.BoxGeometry(0.041, 0.041, len), m.galvanised, 'rail');
  rail.position.copy(c);
  rail.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.normalize());
  g.add(rail);
  for (const t of [0.04, len - 0.04]) {
    const p = a.clone().addScaledVector(dir, t), top = ceilingAt(p.x);
    g.add(ring(new THREE.Vector3(p.x, (top + p.y) / 2, p.z), UP, 0.005, top - p.y, m.chrome, 6));
  }
  return g;
}

/** Corrugated flex duct: a tube along a smooth curve; texture v runs along it. */
function flexDuct(points: V3[], r: number, m: TroneMaterials) {
  const curve = new THREE.CatmullRomCurve3(points.map(v), false, 'centripetal');
  const geo = new THREE.TubeGeometry(curve, 80, r, 16, false);
  const len = curve.getLength();
  const uv = geo.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getY(i), uv.getX(i) * len / 0.025);
  return mesh(geo, m.flexDuct.clone(), 'flex-duct');
}

/**
 * Pipe, duct and drainage networks from networks.json (centrelines measured in the fused LiDAR depth
 * frames where visible, else traced from photos); check-networks.py reports scan deviation and pipe counts.
 */
export function buildNetworks(m: TroneMaterials) {
  const g = new THREE.Group();
  g.name = 'pipework';
  const mats: Record<string, THREE.Material> = {
    galvanised: m.galvanised, brownPvc: m.darkPvc, greyPvc: m.greyPvc, creamPvc: m.whitePipe,
    whiteInsulated: m.whitePipe, blackPipe: m.blackPipe, blackFoam: m.blackFoam, steel: m.chrome,
  };
  for (const run of networks.runs as unknown as Run[]) {
    g.add(run.kind === 'duct' ? flexDuct(run.points, run.d / 2, m) : buildRun(run, mats[run.material], m));
  }
  for (const f of networks.fittings as unknown as Fitting[]) g.add(buildFitting(f, mats[f.material], m));
  for (const rail of networks.rails as unknown as { from: V3; to: V3 }[]) g.add(buildRail(rail.from, rail.to, m));
  return g;
}
