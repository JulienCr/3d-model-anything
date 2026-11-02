import * as THREE from 'three';
import room from '../../data/demos/atrium/room.json';
import { batchStatic } from '../../packages/room-runtime/batch';
import { box, place } from '../../packages/room-runtime/helpers';
import { loadTextures } from '../../packages/room-runtime/materials';
import { loadScanReference } from '../../packages/room-runtime/reference';
import type { BuiltRoom, RoomLayout, RoomModule } from '../../packages/room-runtime/types';
import { buildCeilingItems, TUBE_X } from './ceiling';
import {
  backpack, basket, bookcase, coatStand, cubeShelf, extinguisher, floorLamp, framed, frameTable, laptop, loungeSeat, poufs, radiator, roundTable, tuftedSofa,
} from './furniture';
import { atriumMaterials, type AtriumMaterials } from './materials';
import { onWall } from '../../packages/room-runtime/walls';
import { buildShell, HEIGHT, insideRoom, POLY, type ShellWall } from './shell';

type Footprint = { category: string; center: [number, number, number]; size: [number, number, number]; rotY: number };

/** Furniture placed from the RoomPlan boxes, the scan top view and photos 44-124; positions are room metres. */
function furniture(m: AtriumMaterials, group: THREE.Group, footprints: Footprint[]) {
  const put = <T extends THREE.Object3D>(obj: T, x: number, z: number, rotY = 0, category?: string, size?: [number, number]) => {
    group.add(place(obj, [x, 0, z], rotY));
    if (category && size) footprints.push({ category, center: [x, 0.4, z], size: [size[0], 0.8, size[1]], rotY });
    return obj;
  };
  const seat = (w: number, d: number, back: number, shell: THREE.Material) => loungeSeat({ w, d, back, shell }, m);
  put(seat(1.5, 0.85, 0.82, m.creamFabric), 0.9, -1.4, 1.588, 'sofa', [1.5, 0.85]);
  put(seat(1.55, 0.82, 0.82, m.creamFabric), 3.21, -1.36, -1.118, 'sofa', [1.55, 0.82]);
  put(seat(1.45, 0.8, 0.82, m.redFabric), -1.447, -1.549, -1.027, 'sofa', [1.45, 0.8]);
  put(seat(0.82, 0.85, 0.8, m.greyFabric), -3.72, -1.448, 1.086, 'chair', [0.82, 0.85]);
  put(tuftedSofa(1.6, 0.82, m), -7.39, 1.136, 1.588, 'sofa', [1.6, 0.82]);

  put(frameTable(1.25, 0.55, 0.44, m), 2.016, -1.242, 1.588, 'table', [1.25, 0.55]);
  put(roundTable(0.78, 0.43, m), -2.2, -1.05, 0, 'table', [0.78, 0.78]);
  put(roundTable(0.6, 0.4, m), -2.85, -0.78, 0, 'table', [0.6, 0.6]);
  put(roundTable(0.6, 0.44, m), -6.5, 0.72, 0, 'table', [0.6, 0.6]);
  put(roundTable(0.62, 0.42, m), -6.5, 1.36, 0, 'table', [0.62, 0.62]);
  const lap = (x: number, z: number, y: number, r: number) => group.add(place(laptop(m), [x, y, z], r));
  lap(-6.5, 0.72, 0.44, 1.2); lap(-6.5, 1.38, 0.42, 1.9);

  const orange = '#d8441c', blue = '#1f6fb5', teal = '#1c7e9e', plum = '#5b1b3a';
  group.add(poufs([
    [-3.6, 0.05, 0.3, orange], [-2.6, 0.2, 0.5, orange], [-1.7, -0.4, 0.2, orange], [1.2, -0.15, 0, orange], [1.8, -2.3, 0.4, orange],
    [2.05, -0.1, 0.7, blue], [-3.0, -2.2, 0, blue], [-2.5, -2.2, 0.1, teal],
    [2.75, -0.25, 0.7, plum], [-5.5, 0.95, 0.35, plum], [-6.0, 1.75, 0.1, plum],
  ], m));

  put(bookcase(m), -0.46, -2.28, 0, 'storage', [0.98, 0.34]);
  put(cubeShelf(m), -1.72, 2.68, Math.PI, 'storage', [1.9, 0.36]);
  put(basket(m), 0.22, -2.2);
  for (const [x, z] of [[-4.07, 2.55], [3.82, -2.26], [0.65, -2.25], [-7.2, 2.05], [-7.3, -2.1], [5.45, 2.25]] as const) put(floorLamp(m), x, z);
  put(coatStand(m), -5.75, 2.3);
  put(backpack(m), -5.45, 1.85, 0.5);
}

function mountOnWall(walls: ShellWall[], edge: number, along: [number, number], v: number, obj: THREE.Object3D, out = 0) {
  const w = walls[edge];
  const u = (along[0] - w.def.origin.x) * w.def.u.x + (along[1] - w.def.origin.z) * w.def.u.z;
  onWall(obj, w.def, u, v, out);
  w.group.add(obj);
}

/** Prints and fixtures read off the scan elevations and photos 21, 44, 96, 124, 169. */
function wallDecor(m: AtriumMaterials, walls: ShellWall[]) {
  const earth = ['#5a4636', '#b8956a', '#2d2118', '#d9c7a0'];
  const gaudy = ['#e9d9c9', '#d8302b', '#2a4fa0', '#e8908a', '#f2b64a'];
  const anatomy = ['#e3d2ad', '#a14b3a', '#6b6a58', '#c9a66b'];
  const dark = ['#1b1c21', '#556b6a', '#c6b48a'];
  const north = (x: number) => [x, -2.43 - ((x + 7.86) / 15.74) * 0.12] as [number, number];
  mountOnWall(walls, 2, north(-6.8), 1.3, framed(0.5, 0.7, ['#241f1f', '#a8302a', '#d9a441', '#3b6a8c'], 11, m));
  mountOnWall(walls, 2, north(-2.6), 1.42, framed(0.55, 0.72, gaudy, 5, m, false));
  mountOnWall(walls, 2, north(1.25), 1.35, framed(0.5, 0.68, anatomy, 21, m));
  mountOnWall(walls, 2, north(1.8), 1.52, framed(0.22, 0.3, dark, 3, m));
  mountOnWall(walls, 2, north(1.8), 1.12, framed(0.2, 0.28, dark, 8, m));
  mountOnWall(walls, 2, north(2.28), 1.4, framed(0.42, 0.62, anatomy, 14, m));
  mountOnWall(walls, 5, [3.3, 2.84], 1.6, framed(0.4, 0.55, dark, 2, m));
  mountOnWall(walls, 5, [2.85, 2.84], 1.65, framed(0.2, 0.3, dark, 6, m));
  mountOnWall(walls, 5, [2.2, 2.84], 1.5, framed(0.3, 0.5, anatomy, 4, m));
  mountOnWall(walls, 4, [-6.75, 2.5], 1.35, framed(0.45, 0.62, earth, 17, m));
  mountOnWall(walls, 3, [-7.86, 1.0], 1.5, framed(0.3, 0.4, dark, 9, m));
  mountOnWall(walls, 3, [-7.86, 1.75], 1.5, framed(0.3, 0.42, anatomy, 12, m));
  mountOnWall(walls, 6, [4.5, 2.78], 0.3, radiator(m));
  mountOnWall(walls, 2, north(-6.4), 0.3, radiator(m));
  mountOnWall(walls, 2, north(-7.45), 0.4, extinguisher(m));
  for (const [edge, x, z] of [[2, -5.26, -2.46], [1, 7.88, 0.1]] as const) {
    mountOnWall(walls, edge, [x, z], 2.27, box([0.32, 0.13, 0.03], [0, 0, 0], m.exitSign), 0.02);
  }
}

function layoutOf(walls: ShellWall[], objects: Footprint[]): RoomLayout {
  return {
    note: 'Floor outline measured on the LiDAR mesh; openings from RoomPlan and the scan elevations.',
    polygon: POLY,
    walls: walls.map(({ def, specs }) => ({
      a: [def.origin.x, def.origin.z],
      b: [def.origin.x + def.u.x * def.length, def.origin.z + def.u.z * def.length],
      u: [def.u.x, def.u.z],
      length: def.length,
      openings: specs.map(s => ({ kind: s.kind === 'door' ? 'door' : 'opening', u0: s.u0, u1: s.u1 })),
    })),
    objects,
  };
}

function lighting() {
  const lights = new THREE.Group();
  lights.name = 'atrium-lights';
  lights.add(new THREE.HemisphereLight('#ffffff', '#d2c9bc', 1.0));
  TUBE_X.forEach((x, i) => {
    const spot = new THREE.SpotLight('#fff3e2', 16, 0, THREE.MathUtils.degToRad(80), 1, 2);
    spot.position.set(x, HEIGHT - 0.3, 0.08);
    spot.target.position.set(x, 0, 0.08);
    if (i === 1 || i === 3) {
      spot.castShadow = true;
      spot.shadow.mapSize.set(1024, 1024);
      spot.shadow.camera.near = 0.1;
      spot.shadow.camera.far = 6;
      spot.shadow.bias = -0.0006;
      spot.shadow.normalBias = 0.02;
      spot.shadow.radius = 6;
    }
    lights.add(spot, spot.target);
  });
  return lights;
}

export const atrium: RoomModule = {
  id: 'atrium', label: 'Atrium',
  async build(): Promise<BuiltRoom> {
    await loadTextures();
    const m = await atriumMaterials();
    const group = new THREE.Group();
    group.name = 'atrium';
    const shell = buildShell(m);
    const ceiling = new THREE.Group();
    ceiling.name = 'ceiling';
    ceiling.add(shell.ceiling, buildCeilingItems(m));
    group.add(shell.floor, ceiling, ...shell.walls.map(w => w.group));
    const footprints: Footprint[] = [];
    furniture(m, group, footprints);
    wallDecor(m, shell.walls);

    let optimized: ReturnType<typeof batchStatic> | undefined;
    return {
      layout: layoutOf(shell.walls, footprints),
      group, lights: lighting(),
      optimize(opts) {
        optimized ??= batchStatic(group, [...shell.walls.map(w => w.group), ceiling], { mergeDynamic: !!opts?.static });
        return optimized;
      },
      walls: shell.walls.map(({ def, group: g }) => ({ def, group: g })),
      ceiling: [ceiling],
      wallTop: HEIGHT,
      exposure: 1.1,
      render: { toneMapping: THREE.NeutralToneMapping, envSize: 256, ao: { radius: 0.4, intensity: 0.8, halfRes: true } },
      walkable: (x, z) => [[0, 0], [.2, 0], [-.2, 0], [0, .2], [0, -.2]].every(([dx, dz]) => insideRoom(x + dx, z + dz)),
      camera: { orbit: [0, 11, 9.5], target: [0, 0.8, 0.2], walk: [4.6, 1.6, 1.2], walkLook: [-3, 1.3, -1] },
      environment: { center: [0, 1.2, 0.2], intensity: 0.8 },
      loadReference: onProgress => loadScanReference('/atrium/scan.glb', onProgress),
      referenceAvailable: true,
      referenceKind: 'scan',
      photos: room.photos.map((p, i) => ({
        url: `/atrium/photos/${String(i).padStart(4, '0')}.jpg`,
        position: p.position, forward: p.forward, up: p.up, fovY: p.fovY, aspect: p.aspect,
      })),
    };
  },
};

