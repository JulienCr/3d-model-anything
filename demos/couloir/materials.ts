import * as THREE from 'three';
import { materials as runtimeMaterials, nameMaterials } from '../../packages/room-runtime/materials';

const loader = new THREE.TextureLoader();

/** Own-photo textures live in the git-ignored data/web-assets/couloir; a missing file falls back to a flat colour. */
async function loadOptional(path: string, repeat: [number, number], color = true) {
  try {
    const t = await loader.loadAsync(path);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(...repeat);
    t.anisotropy = 8;
    if (color) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  } catch {
    return null;
  }
}

function canvasTexture(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  draw(canvas.getContext('2d')!);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

/** Green running-man exit sign face (arrow + figure), drawn in code. */
function exitSignTexture(arrow: 'down' | 'right') {
  return canvasTexture(256, 96, ctx => {
    ctx.fillStyle = '#0d9a5c'; ctx.fillRect(0, 0, 256, 96);
    ctx.fillStyle = '#f4fff8';
    ctx.fillRect(10, 10, 76, 76); ctx.fillRect(170, 10, 76, 76);
    ctx.fillStyle = '#0d9a5c';
    ctx.beginPath();
    if (arrow === 'down') { ctx.moveTo(48, 78); ctx.lineTo(22, 46); ctx.lineTo(38, 46); ctx.lineTo(38, 18); ctx.lineTo(58, 18); ctx.lineTo(58, 46); ctx.lineTo(74, 46); }
    else { ctx.moveTo(78, 48); ctx.lineTo(46, 22); ctx.lineTo(46, 38); ctx.lineTo(18, 38); ctx.lineTo(18, 58); ctx.lineTo(46, 58); ctx.lineTo(46, 74); }
    ctx.closePath(); ctx.fill();
    // running figure
    ctx.lineWidth = 9; ctx.lineCap = 'round'; ctx.strokeStyle = '#0d9a5c';
    ctx.beginPath(); ctx.arc(214, 24, 7, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.moveTo(208, 36); ctx.lineTo(198, 58); ctx.lineTo(184, 78); ctx.moveTo(198, 58); ctx.lineTo(214, 66); ctx.lineTo(220, 82);
    ctx.moveTo(206, 40); ctx.lineTo(190, 46); ctx.moveTo(206, 40); ctx.lineTo(226, 50); ctx.stroke();
    ctx.fillStyle = '#f4fff8'; ctx.fillRect(100, 20, 56, 56);
    ctx.fillStyle = '#0d9a5c'; ctx.fillRect(106, 26, 44, 44);
  });
}

/** Small door plaque with a round pictogram and a label. */
function plaqueTexture(text: string, bg: string, fg: string) {
  return canvasTexture(256, 64, ctx => {
    ctx.fillStyle = bg; ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = fg; ctx.beginPath(); ctx.arc(30, 32, 18, 0, 7); ctx.fill();
    ctx.font = 'bold 26px sans-serif'; ctx.textBaseline = 'middle'; ctx.fillText(text, 58, 34);
  });
}

/** Evacuation plan: white sheet with a green title band and a schematic floor plan. */
function evacuationPlanTexture() {
  return canvasTexture(192, 256, ctx => {
    ctx.fillStyle = '#f6f6f2'; ctx.fillRect(0, 0, 192, 256);
    ctx.fillStyle = '#1e8f4e'; ctx.fillRect(0, 0, 192, 22);
    ctx.strokeStyle = '#555'; ctx.lineWidth = 2; ctx.strokeRect(16, 40, 160, 120);
    ctx.fillStyle = '#7fc69a'; ctx.fillRect(30, 60, 60, 30); ctx.fillRect(100, 100, 60, 40);
    ctx.fillStyle = '#d33'; for (let i = 0; i < 6; i++) ctx.fillRect(24 + i * 26, 172, 14, 10);
    ctx.fillStyle = '#e4a33a'; ctx.fillRect(16, 200, 160, 14);
    ctx.fillStyle = '#999'; for (let i = 0; i < 3; i++) ctx.fillRect(16, 222 + i * 9, 120, 4);
  });
}

export async function couloirMaterials() {
  const [concreteMap, floorMap, woolMap, blockMap, rawMap] = await Promise.all([
    loadOptional('/couloir/textures/ceiling.jpg', [1 / 1.2, 1 / 1.2]),
    loadOptional('/couloir/textures/floor.jpg', [1 / 1.5, 1 / 1.5]),
    loadOptional('/couloir/textures/woodwool.jpg', [1 / 0.6, 1 / 0.6]),
    loadOptional('/couloir/textures/blocks.jpg', [1 / 0.75, 1 / 0.8]),
    loadOptional('/couloir/textures/raw-concrete.jpg', [1 / 0.6, 1 / 1.2]),
  ]);
  const rt = runtimeMaterials();
  const solid = (color: string, roughness = 0.6, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
  const emissive = (color: string, glow: string, intensity: number) => new THREE.MeshStandardMaterial({ color, emissive: glow, emissiveIntensity: intensity, roughness: 0.4 });
  return nameMaterials({
    paint: new THREE.MeshStandardMaterial({ color: '#f1eee9', roughness: 0.92, normalMap: rt.wall.normalMap, normalScale: new THREE.Vector2(0.06, 0.06) }),
    paintWarm: new THREE.MeshStandardMaterial({ color: '#efe3cf', roughness: 0.92 }),
    reveal: solid('#ece9e3', 0.8),
    floor: new THREE.MeshStandardMaterial({ color: floorMap ? '#ffffff' : '#aaa69d', map: floorMap, roughness: 0.85 }),
    ceiling: new THREE.MeshStandardMaterial({ color: concreteMap ? '#ffffff' : '#8f8f8c', map: concreteMap, roughness: 0.95, side: THREE.DoubleSide }),
    concrete: new THREE.MeshStandardMaterial({ color: rawMap ? '#ffffff' : '#a59c8c', map: rawMap, roughness: 0.95 }),
    beam: new THREE.MeshStandardMaterial({ color: concreteMap ? '#ffffff' : '#9a9a97', map: concreteMap, roughness: 0.95 }),
    blocks: new THREE.MeshStandardMaterial({ color: blockMap ? '#ffffff' : '#8a8a86', map: blockMap, roughness: 0.95 }),
    woodWool: new THREE.MeshStandardMaterial({ color: woolMap ? '#ffffff' : '#b08a58', map: woolMap, roughness: 1, side: THREE.DoubleSide }),
    mauve: solid('#9a8380', 0.55),
    mauveFrame: solid('#8c7572', 0.5),
    blackDoor: solid('#24272b', 0.55, 0.2),
    whiteDoor: solid('#eceae6', 0.5),
    steelFrame: solid('#2a2c30', 0.5, 0.4),
    sealant: solid('#8c5444', 0.7),
    void: new THREE.MeshStandardMaterial({ color: '#25262c', emissive: '#0a0c14', roughness: 1, side: THREE.BackSide }),
    galvanised: solid('#b4b8bb', 0.35, 0.8),
    conduit: solid('#e9e9e6', 0.5),
    tube: emissive('#ffffff', '#f2f6ff', 2.6),
    lampBody: solid('#d9dadc', 0.5),
    lampCap: solid('#1d1f24', 0.6),
    red: solid('#c4161c', 0.45),
    darkGrey: solid('#3a3c40', 0.6),
    chrome: solid('#c9ccce', 0.25, 1),
    exitRight: new THREE.MeshStandardMaterial({ map: exitSignTexture('right'), emissive: '#ffffff', emissiveMap: exitSignTexture('right'), emissiveIntensity: 0.9, roughness: 0.4 }),
    exitDown: new THREE.MeshStandardMaterial({ map: exitSignTexture('down'), emissive: '#ffffff', emissiveMap: exitSignTexture('down'), emissiveIntensity: 0.9, roughness: 0.4 }),
    signInterdit: new THREE.MeshStandardMaterial({ map: plaqueTexture('ACCÈS INTERDIT', '#c9cbcc', '#1c1c1c'), roughness: 0.4, metalness: 0.3 }),
    signToilets: new THREE.MeshStandardMaterial({ map: plaqueTexture('TOILETTES', '#e8eaf0', '#1f3c8a'), roughness: 0.4 }),
    evacPlan: new THREE.MeshStandardMaterial({ map: evacuationPlanTexture(), roughness: 0.3 }),
    cardboard: solid('#a8865f', 0.9),
    cardboardDark: solid('#8a6c4c', 0.9),
    pipeBlack: solid('#1e1f21', 0.5),
    pipeWhite: solid('#e6e6e2', 0.5),
    bucketGrey: solid('#5e6266', 0.6),
    solid,
  });
}

export type CouloirMaterials = Awaited<ReturnType<typeof couloirMaterials>>;
