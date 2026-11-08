import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/** Load a photo-coloured scan GLB as an unlit reference layer: its baked vertex colours are final. */
export async function loadScanReference(url: string, onProgress?: (p: number) => void) {
  const gltf = await new GLTFLoader().loadAsync(url, e => e.total && onProgress?.(e.loaded / e.total));
  gltf.scene.traverse(o => {
    if (o instanceof THREE.Mesh) o.material = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false, side: THREE.DoubleSide });
  });
  return gltf.scene;
}
