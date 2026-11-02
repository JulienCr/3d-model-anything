import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import room from '../../data/demos/couloir/room.json';
import { batchStatic } from '../../packages/room-runtime/batch';
import { loadTextures } from '../../packages/room-runtime/materials';
import type { BuiltRoom, RoomLayout, RoomModule } from '../../packages/room-runtime/types';
import { buildFixtures, lighting } from './fixtures';
import { couloirMaterials } from './materials';
import { buildShell, insideRoom, POLY, type ShellWall } from './shell';

/** LiDAR mesh coloured from the photos (data/web-assets/couloir/scan.glb); unlit, as the baked colours are final. */
async function loadScan(onProgress?: (p: number) => void) {
  const gltf = await new GLTFLoader().loadAsync('/couloir/scan.glb', e => e.total && onProgress?.(e.loaded / e.total));
  gltf.scene.traverse(o => {
    if (o instanceof THREE.Mesh) o.material = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false, side: THREE.DoubleSide });
  });
  return gltf.scene;
}

function layoutOf(walls: ShellWall[]): RoomLayout {
  return {
    note: 'Floor outline, openings and ceiling steps measured on the LiDAR mesh; RoomPlan used as a hint only.',
    polygon: POLY,
    walls: walls.map(({ def, specs }) => ({
      a: [def.origin.x, def.origin.z],
      b: [def.origin.x + def.u.x * def.length, def.origin.z + def.u.z * def.length],
      u: [def.u.x, def.u.z],
      length: def.length,
      openings: specs.map(s => ({ kind: 'door', u0: s.u0, u1: s.u1 })),
    })),
    objects: [],
  };
}

export const couloir: RoomModule = {
  id: 'couloir', label: 'Couloir',
  async build(): Promise<BuiltRoom> {
    await loadTextures();
    const m = await couloirMaterials();
    const group = new THREE.Group();
    group.name = 'couloir';
    const shell = buildShell(m);
    const ceiling = new THREE.Group();
    ceiling.name = 'ceiling';
    ceiling.add(shell.ceiling);
    group.add(shell.floor, ceiling, ...shell.walls.map(w => w.group));
    buildFixtures(m, shell.walls, ceiling, group);

    let optimized: ReturnType<typeof batchStatic> | undefined;
    return {
      layout: layoutOf(shell.walls),
      group, lights: lighting(),
      optimize(opts) {
        optimized ??= batchStatic(group, [...shell.walls.map(w => w.group), ceiling], { mergeDynamic: !!opts?.static });
        return optimized;
      },
      walls: shell.walls.map(({ def, group: g }) => ({ def, group: g })),
      ceiling: [ceiling],
      wallTop: 2.5,
      exposure: 1.0,
      render: { toneMapping: THREE.NeutralToneMapping, envSize: 256, ao: { radius: 0.35, intensity: 0.8, halfRes: true } },
      walkable: (x, z) => [[0, 0], [.2, 0], [-.2, 0], [0, .2], [0, -.2]].every(([dx, dz]) => insideRoom(x + dx, z + dz)),
      camera: { orbit: [1.5, 11, 8], target: [0, 0.8, 0], walk: [3.9, 1.6, -0.15], walkLook: [-3, 1.3, -0.2] },
      environment: { center: [0, 1.2, -0.15], intensity: 0.7 },
      loadReference: loadScan,
      referenceAvailable: true,
      referenceKind: 'scan',
      photos: room.photos.map((p, i) => ({
        url: `/couloir/photos/${String(i).padStart(4, '0')}.jpg`,
        position: p.position, forward: p.forward, up: p.up, fovY: p.fovY, aspect: p.aspect,
      })),
    };
  },
};
