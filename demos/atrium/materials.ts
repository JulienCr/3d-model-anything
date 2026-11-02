import * as THREE from 'three';
import { loadOptionalTexture, materials as runtimeMaterials, textured, nameMaterials } from '../../packages/room-runtime/materials';

function canvasTexture(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  draw(canvas.getContext('2d')!);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** Button-tufted back of the yellow sofa: diamond grid of dimples, one tile = 0.2 m. */
function tuftTexture() {
  const t = canvasTexture(128, 128, ctx => {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 128, 128);
    ctx.strokeStyle = 'rgba(0,0,0,.28)'; ctx.lineWidth = 3;
    for (const [a, b, c, d] of [[0, 0, 64, 64], [64, 64, 128, 128], [128, 0, 64, 64], [64, 64, 0, 128], [0, 64, 64, 0], [64, 0, 128, 64], [128, 64, 64, 128], [64, 128, 0, 64]]) {
      ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(0,0,0,.4)';
    for (const [x, y] of [[0, 0], [128, 0], [0, 128], [128, 128], [64, 64]]) { ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill(); }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

/** Slotted cable-tray metal: one tile = 0.1 m with a dark slot. */
function perforationTexture() {
  const t = canvasTexture(64, 64, ctx => {
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, 64, 64);
    ctx.fillStyle = '#3a3d40'; ctx.beginPath(); ctx.roundRect(14, 24, 36, 16, 8); ctx.fill();
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(10, 10);
  return t;
}

/** Flat poster art: a few coloured blocks so frames read as prints, not mirrors. */
export function posterTexture(palette: string[], seed: number) {
  return canvasTexture(128, 160, ctx => {
    let s = seed;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    ctx.fillStyle = palette[0]; ctx.fillRect(0, 0, 128, 160);
    for (let i = 0; i < 9; i++) {
      ctx.fillStyle = palette[1 + Math.floor(rnd() * (palette.length - 1))];
      ctx.globalAlpha = 0.55 + rnd() * 0.45;
      const w = 20 + rnd() * 70, h = 20 + rnd() * 70;
      ctx.fillRect(rnd() * (128 - w), rnd() * (160 - h), w, h);
    }
    ctx.globalAlpha = 1;
  });
}

export async function atriumMaterials() {
  // own-photo textures in the git-ignored data/web-assets/atrium; a missing file falls back to a flat colour
  const [ceilingMap, floorMap, oakMap] = await Promise.all([
    loadOptionalTexture('/atrium/textures/ceiling.jpg', [1 / 0.9, 1 / 0.46]),
    loadOptionalTexture('/atrium/textures/floor.jpg', [0.9, 0.9]),
    loadOptionalTexture('/atrium/textures/oak-light.jpg', [1.2, 1.2]),
  ]);
  // Planks run along the long axis (x): the maps are laid out along v, so turn them a quarter.
  const floorNormal = textured({ color: '#fff', set: 'laminate', repeat: 0.9 }).normalMap!;
  for (const t of [floorMap, floorNormal]) if (t) { t.rotation = Math.PI / 2; t.center.set(0.5, 0.5); }
  const rt = runtimeMaterials();
  const solid = (color: string, roughness = 0.6, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
  const wood = (color: string) => textured({ color, set: 'oak', repeat: 1.8, roughness: 0.55 });
  const fabric = (color: string, repeat = 3, normalScale = 1) => textured({ color, set: 'wool', repeat, normalScale, roughness: 1 });
  const perforated = perforationTexture();
  const tuft = tuftTexture();
  tuft.repeat.set(5, 5);

  return nameMaterials({
    paint: new THREE.MeshStandardMaterial({ color: '#fcfbf8', roughness: 0.92, normalMap: rt.wall.normalMap, normalScale: new THREE.Vector2(0.08, 0.08) }),
    reveal: solid('#a99d98', 0.55),
    baseboard: solid('#d6d4cf', 0.5),
    door: solid('#a99a96', 0.5),
    doorSteel: solid('#aab3bd', 0.4, 0.3),
    floor: new THREE.MeshStandardMaterial({
      color: '#ffffff', map: floorMap, roughness: 0.62,
      normalMap: floorNormal, normalScale: new THREE.Vector2(0.5, 0.5),
    }),
    ceiling: new THREE.MeshStandardMaterial({ color: '#ffffff', map: ceilingMap, roughness: 1, side: THREE.DoubleSide }),
    duct: solid('#f3f2ee', 0.7),
    tray: new THREE.MeshStandardMaterial({ color: '#c4c8ca', roughness: 0.5, metalness: 0.3, map: perforated }),
    clamp: solid('#dcdcd8', 0.5),
    tube: new THREE.MeshStandardMaterial({ color: '#fff', emissive: '#fff6e8', emissiveIntensity: 2.2, roughness: 0.4 }),
    exitSign: new THREE.MeshStandardMaterial({ color: '#1c8a4a', emissive: '#1fbf63', emissiveIntensity: 1.6 }),
    void: new THREE.MeshStandardMaterial({ color: '#25262c', emissive: '#0a0c14', roughness: 1, side: THREE.BackSide }),
    voidLit: new THREE.MeshStandardMaterial({ color: '#8d8a85', emissive: '#4a4843', roughness: 1, side: THREE.BackSide }),
    voidBlue: new THREE.MeshStandardMaterial({ color: '#323a66', emissive: '#27327a', emissiveIntensity: 0.9, roughness: 1, side: THREE.BackSide }),
    steel: solid('#1b1b1c', 0.5, 0.7),
    chrome: solid('#c9ccce', 0.25, 1),
    brass: solid('#b9945a', 0.35, 0.9),
    oakLeg: wood('#d1b48a'),
    oakTop: Object.assign(textured({ color: oakMap ? '#ffffff' : '#ccb28c', set: 'oak', repeat: 1.2, roughness: 0.5 }), oakMap ? { map: oakMap } : {}),
    walnut: solid('#34211a', 0.55),
    creamFabric: fabric('#c9beae', 3.5, 1.4),
    redFabric: fabric('#c8402a', 3, 0.8),
    greyFabric: fabric('#4b4d51', 3, 1),
    yellowFabric: fabric('#e0aa1c', 3, 0.5),
    poufFabric: fabric('#ffffff', 3, 0.6),
    tuftedYellow: new THREE.MeshStandardMaterial({ color: '#e0aa1c', roughness: 0.85, bumpMap: tuft, bumpScale: 1.4 }),
    tableTop: solid('#cdc7ba', 0.5),
    lampShade: new THREE.MeshStandardMaterial({ color: '#cdc2ad', emissive: '#ffe9c4', emissiveIntensity: 0.12, roughness: 0.95, side: THREE.DoubleSide }),
    radiator: solid('#e6e8e8', 0.45),
    white: solid('#f1f1ee', 0.45),
    leaf: new THREE.MeshStandardMaterial({ color: '#c9742a', roughness: 0.9, side: THREE.DoubleSide }),
    solid,
    fabric,
    wood,
  });
}

export type AtriumMaterials = Awaited<ReturnType<typeof atriumMaterials>>;
