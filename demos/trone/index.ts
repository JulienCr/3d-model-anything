import * as THREE from 'three';
import room from '../../data/demos/trone/room.json';
import { batchStatic } from '../../packages/room-runtime/batch';
import { loadTextures } from '../../packages/room-runtime/materials';
import { loadScanReference } from '../../packages/room-runtime/reference';
import type { BuiltRoom, RoomLayout, RoomModule } from '../../packages/room-runtime/types';
import { buildClutter } from './clutter';
import { troneMaterials } from './materials';
import { buildServices, PILLAR } from './services';
import { buildShell, FLOOR_Y, HEIGHT, insideRoom, POLY, type ShellWall } from './shell';


function layoutOf(walls: ShellWall[]): RoomLayout {
  return {
    note: 'Floor outline, door and ceiling steps measured on the LiDAR mesh; RoomPlan used as a hint only.',
    polygon: POLY,
    walls: walls.map(({ def, door }) => ({
      a: [def.origin.x, def.origin.z],
      b: [def.origin.x + def.u.x * def.length, def.origin.z + def.u.z * def.length],
      u: [def.u.x, def.u.z],
      length: def.length,
      openings: door ? [{ kind: 'door', u0: door.u0, u1: door.u1 }] : [],
    })),
    objects: [],
  };
}

/** Light positions triangulated from saturated blobs in the photos (lights_from_photos.py, then per-fixture rays). */
function lighting() {
  const lights = new THREE.Group();
  lights.name = 'trone-lights';
  // warm ground term: the cream pipes' undersides catch light bounced off the dusty floor
  lights.add(new THREE.HemisphereLight('#f3ede2', '#a8977c', 1.15));
  const spot = (color: string, cd: number, pos: THREE.Vector3Tuple, target: THREE.Vector3Tuple, angle: number, shadow: boolean) => {
    const s = new THREE.SpotLight(color, cd, 0, THREE.MathUtils.degToRad(angle), 1, 2);
    s.position.set(...pos);
    s.target.position.set(...target);
    if (shadow) {
      s.castShadow = true;
      s.shadow.mapSize.set(1024, 1024);
      s.shadow.camera.near = 0.1;
      s.shadow.camera.far = 14;
      s.shadow.bias = -0.0005;
      s.shadow.normalBias = 0.02;
    }
    lights.add(s, s.target);
  };
  // cool LED site floodlights: the west corner one and the one clamped on the drain pipes
  spot('#d8e3ff', 45, [-8.5, 1.88, -2.3], [-6, 0.4, 0.2], 62, true);
  spot('#d8e3ff', 35, [-0.85, 1.9, 1.32], [-4, 0.3, 0.4], 62, true);
  // warm household bulbs in the floor lamps
  const point = (color: string, cd: number, pos: THREE.Vector3Tuple) => {
    const p = new THREE.PointLight(color, cd, 0, 2);
    p.position.set(...pos);
    lights.add(p);
  };
  // lamp bases ray-cast from photos 180, 209, 287
  point('#ff9f50', 34, [2.64, 1.62, -1.76]);
  point('#ffa95e', 40, [5.94, 1.6, -1.3]);
  return lights;
}

export const trone: RoomModule = {
  id: 'trone', label: 'Trone',
  async build(): Promise<BuiltRoom> {
    await loadTextures();
    const m = await troneMaterials();
    const group = new THREE.Group();
    group.name = 'trone';
    const shell = buildShell(m);
    group.add(shell.floor, shell.ceiling, ...shell.walls.map(w => w.group));
    group.add(buildServices(m, shell.walls), buildClutter(m));

    let optimized: ReturnType<typeof batchStatic> | undefined;
    return {
      layout: layoutOf(shell.walls),
      group, lights: lighting(),
      optimize(opts) {
        optimized ??= batchStatic(group, [...shell.walls.map(w => w.group), shell.ceiling], { mergeDynamic: !!opts?.static });
        return optimized;
      },
      walls: shell.walls.map(({ def, group: g }) => ({ def, group: g })),
      ceiling: [shell.ceiling],
      wallTop: HEIGHT,
      exposure: 1.0,
      render: { toneMapping: THREE.NeutralToneMapping, envSize: 256, envNeutral: true, ao: { radius: 0.4, intensity: 0.8, halfRes: true } },
      walkable: (x, z) => [[0, 0], [.2, 0], [-.2, 0], [0, .2], [0, -.2]].every(([dx, dz]) => insideRoom(x + dx, z + dz))
        && !(x > PILLAR.x0 - 0.2 && x < PILLAR.x1 + 0.2 && z > PILLAR.z0 - 0.2 && z < PILLAR.z1 + 0.2),
      camera: { orbit: [0, 12, 10], target: [0, 0.8, 0], walk: [-6, 1.6, 0.2], walkLook: [3, 1.3, 0] },
      environment: { center: [0, 1.2, 0], intensity: 0.08 },
      loadReference: onProgress => loadScanReference('/trone/scan.glb', onProgress),
      referenceAvailable: true,
      referenceKind: 'scan',
      photos: room.photos.map((p, i) => ({
        url: `/trone/photos/${String(i).padStart(4, '0')}.jpg`,
        position: [p.position[0], p.position[1] - FLOOR_Y, p.position[2]], forward: p.forward, up: p.up, fovY: p.fovY, aspect: p.aspect,
      })),
    };
  },
};
