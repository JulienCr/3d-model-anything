import * as THREE from 'three';

// Base colors are the scan's baked colors brightened toward true albedo
export const COLORS = {
  wall: '#e6d6ba',
  trim: '#eee6d6',
  ceiling: '#efe9de',
  carpet: '#b6a793',
  leather: '#6a4632',
  ottoman: '#dccbb4',
  clubChair: '#b09772',
  greyChair: '#6e655b',
  console: '#ece9e2',
  mantel: '#e6d5b8',
  woodTable: '#a3743f',
};

const loader = new THREE.TextureLoader();
const cache = new Map<string, THREE.Texture>();
let anisotropy = 8;

export function setMaxAnisotropy(value: number) {
  anisotropy = value;
}

const SETS = ['carpet', 'leather', 'linen', 'wool', 'oak', 'woodfloor', 'laminate'];

let texturesReady: Promise<void> | undefined;

/** Share completed and in-flight loads across room builds; failed loads can retry. */
export function loadTextures(): Promise<void> {
  return texturesReady ??= preloadTextures().catch(error => {
    texturesReady = undefined;
    throw error;
  });
}

async function preloadTextures() {
  const files: [string, boolean][] = SETS.flatMap((set): [string, boolean][] => [
    [`${set}_diff.jpg`, true],
    [`${set}_nor.jpg`, false],
    [`${set}_rough.jpg`, false],
  ]);
  files.push(['plaster_nor.jpg', false]);
  const results = await Promise.allSettled(
    files.filter(([path, color]) => !cache.has(`${path}:${color}`)).map(async ([path, color]) => {
      const t = await loader.loadAsync(`/textures/${path}`);
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = anisotropy;
      if (color) t.colorSpace = THREE.SRGBColorSpace;
      cache.set(`${path}:${color}`, t);
    }),
  );
  // Finish pending successes before exposing a retry after any failure.
  const failed = results.find(result => result.status === 'rejected');
  if (failed?.status === 'rejected') throw failed.reason;
}

function tex(path: string, color = false): THREE.Texture {
  const t = cache.get(`${path}:${color}`);
  if (!t) throw new Error(`texture ${path} not preloaded (call loadTextures first)`);
  return t;
}

// Each material gets a clone with its own repeat; clones share the image
// source, so they upload once the original finishes loading
function repeated(t: THREE.Texture, repeat: number): THREE.Texture {
  const c = t.clone();
  c.repeat.set(repeat, repeat);
  return c;
}

interface TexturedOptions {
  color: THREE.ColorRepresentation;
  /** texture set name in public/textures (<name>_diff/_nor/_rough.jpg) */
  set: string;
  /** texture repeats per meter, assuming UVs are in meters */
  repeat?: number;
  diffuse?: boolean;
  normalScale?: number;
  roughness?: number;
  side?: THREE.Side;
}

export function textured(o: TexturedOptions): THREE.MeshStandardMaterial {
  const r = o.repeat ?? 1;
  return new THREE.MeshStandardMaterial({
    color: o.color,
    map: o.diffuse === false ? null : repeated(tex(`${o.set}_diff.jpg`, true), r),
    normalMap: repeated(tex(`${o.set}_nor.jpg`), r),
    normalScale: new THREE.Vector2(o.normalScale ?? 1, o.normalScale ?? 1),
    roughnessMap: repeated(tex(`${o.set}_rough.jpg`), r),
    roughness: o.roughness ?? 1,
    side: o.side ?? THREE.FrontSide,
  });
}

/** Load a room-specific texture that may be absent (git-ignored own-photo assets); resolves null when missing. */
export async function loadOptionalTexture(path: string, repeat: [number, number], color = true) {
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

/** Name each material of a palette after its key, so exports and Blender show roles instead of "Material.012". */
export function nameMaterials<T extends Record<string, unknown>>(palette: T): T {
  for (const [key, value] of Object.entries(palette)) if (value instanceof THREE.Material && !value.name) value.name = key;
  return palette;
}

// --- procedural canvas textures -------------------------------------------

function canvasTexture(size: number, draw: (ctx: CanvasRenderingContext2D, size: number) => void) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  draw(canvas.getContext('2d')!, size);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = anisotropy;
  return t;
}

/** The ottoman's beige/pink/grey tartan. One tile ≈ 0.3m. */
export function plaidTexture() {
  return canvasTexture(512, (ctx, s) => {
    ctx.fillStyle = '#e3d3c1';
    ctx.fillRect(0, 0, s, s);
    const bands: [number, number, string, number][] = [
      // offset, width (fraction of tile), color, alpha
      [0.0, 0.16, '#c9a99a', 0.55], // dusty pink wide band
      [0.2, 0.025, '#6f6a5c', 0.8], // dark olive pinstripes
      [0.25, 0.025, '#6f6a5c', 0.8],
      [0.42, 0.1, '#b9ab93', 0.55], // taupe band
      [0.6, 0.012, '#8a4f4a', 0.7], // red hairline
      [0.72, 0.05, '#8d8575', 0.6], // grey band
      [0.86, 0.018, '#6f6a5c', 0.8],
    ];
    for (const [off, w, color, a] of bands) {
      ctx.globalAlpha = a;
      ctx.fillStyle = color;
      ctx.fillRect(off * s, 0, w * s, s); // warp
      ctx.fillRect(0, off * s, s, w * s); // weft
    }
    // woven texture: fine diagonal twill lines
    ctx.globalAlpha = 0.06;
    ctx.strokeStyle = '#000';
    for (let i = -s; i < s; i += 4) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i + s, s);
      ctx.stroke();
    }
  });
}

/** Grey armchair's woven stripes (vertical). */
export function stripeTexture() {
  return canvasTexture(256, (ctx, s) => {
    ctx.fillStyle = '#726a60';
    ctx.fillRect(0, 0, s, s);
    const stripes: [number, number, string][] = [
      [0.05, 0.06, '#4e4841'],
      [0.3, 0.02, '#8f877b'],
      [0.36, 0.02, '#8f877b'],
      [0.6, 0.08, '#57514a'],
      [0.85, 0.03, '#8f877b'],
    ];
    for (const [off, w, c] of stripes) {
      ctx.fillStyle = c;
      ctx.fillRect(off * s, 0, w * s, s);
    }
  });
}

/** Abstract blue/gold painting above the sofa */
export function paintingTexture() {
  return canvasTexture(512, (ctx, s) => {
    const g = ctx.createLinearGradient(0, 0, s, s);
    g.addColorStop(0, '#3d5c8c');
    g.addColorStop(0.5, '#6d8fb8');
    g.addColorStop(1, '#2c4468');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, s, s);
    const rand = mulberry32(7);
    const palette = ['#d9b45a', '#e8e1cf', '#1f3050', '#8fb0d6', '#b98a3e', '#f2efe6'];
    for (let i = 0; i < 900; i++) {
      ctx.globalAlpha = 0.25 + rand() * 0.5;
      ctx.fillStyle = palette[Math.floor(rand() * palette.length)];
      const x = rand() * s, y = rand() * s;
      ctx.beginPath();
      ctx.ellipse(x, y, 2 + rand() * 14, 1 + rand() * 5, rand() * Math.PI, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

/** Rug-like patterned floor cushion near the fireplace */
export function kilimTexture() {
  return canvasTexture(256, (ctx, s) => {
    ctx.fillStyle = '#3a4150';
    ctx.fillRect(0, 0, s, s);
    const rand = mulberry32(3);
    const palette = ['#b07a4f', '#d8c7a8', '#6b2f2c', '#7c8aa0'];
    for (let y = 0; y < s; y += 16) {
      for (let x = (y / 16) % 2 ? 8 : 0; x < s; x += 16) {
        ctx.fillStyle = palette[Math.floor(rand() * palette.length)];
        ctx.beginPath();
        ctx.moveTo(x + 8, y);
        ctx.lineTo(x + 16, y + 8);
        ctx.lineTo(x + 8, y + 16);
        ctx.lineTo(x, y + 8);
        ctx.fill();
      }
    }
  });
}

export function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- shared material palette ------------------------------------------------

let palette: ReturnType<typeof createPalette> | undefined;

export function materials() {
  palette ??= createPalette();
  return palette;
}

function createPalette() {
  const plaster = (color: string) =>
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.92,
      normalMap: repeated(tex('plaster_nor.jpg'), 0.5),
      normalScale: new THREE.Vector2(0.15, 0.15),
    });

  return {
    wall: plaster(COLORS.wall),
    ceiling: plaster(COLORS.ceiling),
    trim: new THREE.MeshStandardMaterial({ color: COLORS.trim, roughness: 0.55, side: THREE.DoubleSide }),
    carpet: textured({ color: COLORS.carpet, set: 'carpet', repeat: 1.2, normalScale: 1.2 }),
    woodFloor: textured({ color: '#ffffff', set: 'woodfloor', repeat: 0.5, roughness: 0.8 }),
    leather: textured({ color: COLORS.leather, set: 'leather', repeat: 1.5, roughness: 1, normalScale: 0.8 }),
    clubChair: textured({ color: COLORS.clubChair, set: 'linen', repeat: 3, normalScale: 1.5 }),
    linenWhite: textured({ color: '#f1ece2', set: 'linen', repeat: 3 }),
    greyChair: new THREE.MeshStandardMaterial({
      map: (() => { const t = stripeTexture(); t.repeat.set(5, 1); return t; })(),
      normalMap: repeated(tex('wool_nor.jpg'), 3),
      roughness: 1,
    }),
    ottoman: new THREE.MeshStandardMaterial({
      map: (() => { const t = plaidTexture(); t.repeat.set(3.3, 3.3); return t; })(),
      normalMap: repeated(tex('linen_nor.jpg'), 3),
      roughness: 0.95,
    }),
    oak: textured({ color: '#c89a68', set: 'oak', repeat: 1.5, roughness: 0.7 }),
    darkWood: textured({ color: '#6b4a32', set: 'oak', repeat: 1.5, roughness: 0.6 }),
    console: new THREE.MeshStandardMaterial({ color: COLORS.console, roughness: 0.5 }),
    mantel: new THREE.MeshStandardMaterial({ color: COLORS.mantel, roughness: 0.75 }),
    brass: new THREE.MeshStandardMaterial({ color: '#c9a45c', metalness: 1, roughness: 0.3 }),
    silver: new THREE.MeshStandardMaterial({ color: '#b8b8b8', metalness: 1, roughness: 0.35 }),
    iron: new THREE.MeshStandardMaterial({ color: '#2b2b2b', metalness: 0.6, roughness: 0.5 }),
    gilt: new THREE.MeshStandardMaterial({ color: '#b08a45', metalness: 0.9, roughness: 0.35 }),
    blackGloss: new THREE.MeshStandardMaterial({ color: '#050505', roughness: 0.12, metalness: 0.2 }),
    blackMatte: new THREE.MeshStandardMaterial({ color: '#111111', roughness: 0.8 }),
    soot: new THREE.MeshStandardMaterial({ color: '#141210', roughness: 1 }),
    glass: new THREE.MeshStandardMaterial({
      color: '#0b1016',
      roughness: 0.05,
      metalness: 0.3,
      transparent: true,
      opacity: 0.85,
    }),
    mirror: new THREE.MeshStandardMaterial({ color: '#c9b893', roughness: 0.06, metalness: 1 }),
    candle: new THREE.MeshStandardMaterial({ color: '#f4efe4', roughness: 0.6, emissive: '#ffcf8a', emissiveIntensity: 0.05 }),
    lampShade: new THREE.MeshStandardMaterial({
      color: '#f5efe2',
      roughness: 0.9,
      emissive: '#ffd59a',
      emissiveIntensity: 0.9,
      side: THREE.DoubleSide,
    }),
    bulbGlow: new THREE.MeshBasicMaterial({ color: '#fff4dc' }),
    potLightTrim: new THREE.MeshStandardMaterial({ color: '#f4f2ee', roughness: 0.4 }),
    painting: new THREE.MeshStandardMaterial({ map: paintingTexture(), roughness: 0.8 }),
    kilim: new THREE.MeshStandardMaterial({ map: kilimTexture(), roughness: 1 }),
  };
}

export type Palette = ReturnType<typeof createPalette>;
