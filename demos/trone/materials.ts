import * as THREE from 'three';
import { loadOptionalTexture, materials as runtimeMaterials, mulberry32, textured, nameMaterials } from '../../packages/room-runtime/materials';

function canvasTexture(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void, repeat: [number, number] = [1, 1]) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  draw(canvas.getContext('2d')!);
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(...repeat);
  t.anisotropy = 8;
  return t;
}

/**
 * Pink fire-rated plasterboard as fitted (photos 0, 12, 131): 1.2 m boards, a feathered white joint band
 * over each seam and a column of filled screw heads every 0.25 m down the middle. One tile = 1.2 x 1.0 m.
 */
function plasterboardTexture() {
  return canvasTexture(480, 400, ctx => {
    ctx.fillStyle = '#c98f87';
    ctx.fillRect(0, 0, 480, 400);
    const rnd = mulberry32(11);
    const joint = ctx.createLinearGradient(0, 0, 480, 0);
    joint.addColorStop(0, 'rgba(236,232,224,1)'); joint.addColorStop(0.07, 'rgba(236,232,224,1)');
    joint.addColorStop(0.11, 'rgba(236,232,224,0)'); joint.addColorStop(0.89, 'rgba(236,232,224,0)');
    joint.addColorStop(0.93, 'rgba(236,232,224,1)'); joint.addColorStop(1, 'rgba(236,232,224,1)');
    ctx.fillStyle = joint;
    ctx.fillRect(0, 0, 480, 400);
    ctx.fillStyle = '#ece8e0';
    for (let y = 50; y < 400; y += 100) {
      ctx.beginPath();
      for (let k = 0; k < 9; k++) {
        const a = (k / 9) * Math.PI * 2, r = 17 + rnd() * 8;
        const x = 240 + Math.cos(a) * r * 0.9, yy = y + Math.sin(a) * r;
        if (k) ctx.lineTo(x, yy); else ctx.moveTo(x, yy);
      }
      ctx.fill();
    }
  }, [1 / 1.2, 1 / 1.0]);
}

/** Bare concrete blockwork, 0.5 x 0.2 m blocks with grey mortar (photos 116, 117). */
function blockTexture() {
  return canvasTexture(256, 205, ctx => {
    const rnd = mulberry32(5);
    ctx.fillStyle = '#5f5a53';
    ctx.fillRect(0, 0, 256, 205);
    for (const [x, row] of [[0, 0], [-128, 1], [128, 1]]) {
      const v = 128 + Math.floor(rnd() * 22);
      ctx.fillStyle = `rgb(${v},${v - 4},${v - 10})`;
      ctx.fillRect(x + 3, row * 102.5 + 3, 256 - 6, 102.5 - 6);
    }
    for (let i = 0; i < 700; i++) {
      ctx.fillStyle = `rgba(60,55,50,${0.1 + rnd() * 0.25})`;
      ctx.fillRect(rnd() * 256, rnd() * 205, 1 + rnd() * 2, 1 + rnd() * 2);
    }
  }, [1 / 0.5, 1 / 0.4]);
}

/** Black moulded faux-brick wall panels leaning on the south wall (photos 76, 79, 99). One tile = 0.25 x 0.13 m. */
function fauxBrickTexture() {
  return canvasTexture(128, 128, ctx => {
    ctx.fillStyle = '#121214';
    ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = '#2a2a2d';
    for (let row = 0; row < 2; row++) {
      const off = row % 2 ? 64 : 0;
      for (let x = -64; x < 128; x += 128) ctx.fillRect(x + off + 4, row * 64 + 4, 120, 56);
    }
  }, [1 / 0.25, 1 / 0.13]);
}

/** Corrugated aluminium flex duct: one ring per texture period along the duct (cylinder v). */
function flexDuctTexture() {
  return canvasTexture(32, 32, ctx => {
    const g = ctx.createLinearGradient(0, 0, 0, 32);
    g.addColorStop(0, '#a9acaf'); g.addColorStop(0.5, '#ffffff'); g.addColorStop(1, '#a9acaf');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 32, 32);
  });
}

/** Chipboard / OSB grain for the leaning boards. One tile = 0.5 m. */
function chipboardTexture() {
  return canvasTexture(256, 256, ctx => {
    const rnd = mulberry32(9);
    ctx.fillStyle = '#c4baa8';
    ctx.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 900; i++) {
      const v = 150 + rnd() * 70;
      ctx.fillStyle = `rgba(${v},${v * 0.9},${v * 0.75},0.55)`;
      ctx.save();
      ctx.translate(rnd() * 256, rnd() * 256); ctx.rotate(rnd() * Math.PI);
      ctx.fillRect(-6 - rnd() * 8, -1.5, 12 + rnd() * 16, 3 + rnd() * 3);
      ctx.restore();
    }
  }, [2, 2]);
}

/** Green emergency-exit pictogram with a white arrow (photo 133). */
function exitSignTexture() {
  return canvasTexture(256, 96, ctx => {
    ctx.fillStyle = '#1f9a52';
    ctx.fillRect(0, 0, 256, 96);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(14, 10, 76, 76);
    ctx.fillStyle = '#1f9a52';
    ctx.beginPath(); ctx.moveTo(28, 40); ctx.lineTo(56, 40); ctx.lineTo(56, 26); ctx.lineTo(80, 48); ctx.lineTo(56, 70); ctx.lineTo(56, 56); ctx.lineTo(28, 56); ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 40px sans-serif';
    ctx.fillText('⇨', 120, 62);
  });
}

/** Soft dark blotches (soot, damp, dirt) for large-scale variation; used as an AO map with its own repeat. */
function stainTexture(seed: number, blots: number, strength: number) {
  return canvasTexture(256, 256, ctx => {
    const rnd = mulberry32(seed);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, 256, 256);
    for (let i = 0; i < blots; i++) {
      const x = rnd() * 256, y = rnd() * 256, r = 12 + rnd() * 60, a = strength * (0.3 + rnd() * 0.7);
      for (const [dx, dy] of [[0, 0], [256, 0], [-256, 0], [0, 256], [0, -256]]) {
        const g = ctx.createRadialGradient(x + dx, y + dy, 0, x + dx, y + dy, r);
        g.addColorStop(0, `rgba(40,36,30,${a})`); g.addColorStop(1, 'rgba(40,36,30,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x + dx - r, y + dy - r, 2 * r, 2 * r);
      }
    }
  });
}

/** Faint grey cast-concrete mottling (photos 166, 167, 220). One tile = 1.5 m. */
function mottleTexture() {
  return canvasTexture(256, 256, ctx => {
    const rnd = mulberry32(17);
    ctx.fillStyle = '#e2e2df';
    ctx.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 400; i++) {
      const v = 150 + rnd() * 90;
      ctx.fillStyle = `rgba(${v},${v},${v - 4},0.1)`;
      ctx.beginPath(); ctx.ellipse(rnd() * 256, rnd() * 256, 3 + rnd() * 18, 2 + rnd() * 10, rnd() * 3, 0, 7); ctx.fill();
    }
  }, [1 / 1.5, 1 / 1.5]);
}

export async function troneMaterials() {
  // own-photo textures in the git-ignored data/web-assets/trone; a missing file falls back to a flat colour
  const [ceilingMap, floorMap, wallMap, pilasterMap] = await Promise.all([
    loadOptionalTexture('/trone/textures/ceiling.jpg', [1 / 1.3, 1 / 0.75]),
    loadOptionalTexture('/trone/textures/floor.jpg', [1 / 1.6, 1 / 1.1]),
    loadOptionalTexture('/trone/textures/wall.jpg', [1 / 2.4, 1 / 2.2]),
    loadOptionalTexture('/trone/textures/pilaster.jpg', [1 / 0.3, 1 / 2.18]),
  ]);
  // pilaster boxes are centred on half height, so shift the photo down to start at the floor
  pilasterMap?.offset.set(0, 0.5);
  const ceilingStain = stainTexture(3, 40, 0.55), floorStain = stainTexture(8, 30, 0.35), wallStain = stainTexture(5, 25, 0.4);
  ceilingStain.repeat.set(1 / 4, 1 / 4); floorStain.repeat.set(1 / 3, 1 / 3); wallStain.repeat.set(1 / 3, 1 / 2.2);
  const mottle = mottleTexture();
  const rt = runtimeMaterials();
  const solid = (color: THREE.ColorRepresentation, roughness = 0.6, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
  const plasterNormal = rt.wall.normalMap;
  const duct = flexDuctTexture();

  return nameMaterials({
    concrete: new THREE.MeshStandardMaterial({
      color: wallMap ? '#dde2e5' : '#d0d4d6', map: wallMap, roughness: 0.95, aoMap: wallStain, aoMapIntensity: 1,
      normalMap: plasterNormal, normalScale: new THREE.Vector2(0.25, 0.25),
    }),
    greyConcrete: new THREE.MeshStandardMaterial({ color: '#c9c8c4', map: mottle, roughness: 0.95, normalMap: plasterNormal, normalScale: new THREE.Vector2(0.4, 0.4) }),
    pillarConcrete: new THREE.MeshStandardMaterial({ color: '#d6c8b0', map: mottle, roughness: 0.95, normalMap: plasterNormal, normalScale: new THREE.Vector2(0.4, 0.4) }),
    pilasterFace: new THREE.MeshStandardMaterial({ color: pilasterMap ? '#fff4e6' : '#b0ada7', map: pilasterMap ?? mottle, roughness: 0.95, normalMap: plasterNormal, normalScale: new THREE.Vector2(0.4, 0.4) }),
    plasterboard: new THREE.MeshStandardMaterial({ color: '#ffffff', map: plasterboardTexture(), roughness: 0.9 }),
    skim: solid('#e6e1d8', 0.9),
    block: new THREE.MeshStandardMaterial({ color: '#ffffff', map: blockTexture(), roughness: 0.95, normalMap: plasterNormal, normalScale: new THREE.Vector2(0.6, 0.6) }),
    floor: new THREE.MeshStandardMaterial({
      color: floorMap ? '#aeaba5' : '#9c9993', map: floorMap, roughness: 0.92,
      normalMap: plasterNormal, normalScale: new THREE.Vector2(0.15, 0.15), aoMap: floorStain, aoMapIntensity: 1,
    }),
    flock: new THREE.MeshStandardMaterial({
      color: ceilingMap ? '#d9d6d0' : '#c9c5bd', map: ceilingMap, roughness: 1, side: THREE.DoubleSide,
      bumpMap: ceilingMap, bumpScale: 1.2, aoMap: ceilingStain, aoMapIntensity: 1,
    }),
    soffit: solid('#b3ada3', 0.95),
    trim: solid('#b9b2a6', 0.8),
    doorLeaf: solid('#ecebe6', 0.45),
    doorFrame: textured({ color: '#d39a5c', set: 'oak', repeat: 2, roughness: 0.6 }),
    steel: solid('#3d3f42', 0.5, 0.6),
    chrome: solid('#c9ccce', 0.25, 1),
    greyPvc: solid('#9a9ea1', 0.45),
    darkPvc: solid('#77726b', 0.5),
    galvanised: solid('#9a9c9b', 0.42, 0.75),
    whitePipe: solid('#e6e3dc', 0.85),
    blackPipe: solid('#202123', 0.5),
    blackFoam: solid('#1d1c1c', 0.85),
    flexDuct: new THREE.MeshStandardMaterial({ color: '#e4e6e8', map: duct, bumpMap: duct, bumpScale: 1, roughness: 0.3, metalness: 0.6 }),
    grille: solid('#2b2b2c', 0.7),
    exitSign: new THREE.MeshStandardMaterial({ map: exitSignTexture(), emissive: '#ffffff', emissiveMap: exitSignTexture(), emissiveIntensity: 1.2 }),
    lampGlow: new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: '#fff7ea', emissiveIntensity: 6 }),
    ledGlow: new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: '#eef4ff', emissiveIntensity: 8 }),
    bulb: new THREE.MeshStandardMaterial({ color: '#fff', emissive: '#ffd9a0', emissiveIntensity: 8 }),
    shade: new THREE.MeshStandardMaterial({ color: '#f5efe2', emissive: '#ffe2b0', emissiveIntensity: 1.4, roughness: 0.9, side: THREE.DoubleSide }),
    lampMetal: solid('#1d1d1e', 0.4, 0.6),
    brass: solid('#b48a4a', 0.35, 0.9),
    chipboard: new THREE.MeshStandardMaterial({ color: '#ffffff', map: chipboardTexture(), roughness: 0.85 }),
    plywood: textured({ color: '#d8c3a0', set: 'oak', repeat: 1.5, roughness: 0.75 }),
    pine: textured({ color: '#e3cda5', set: 'oak', repeat: 2.5, roughness: 0.7 }),
    fauxBrick: new THREE.MeshStandardMaterial({ color: '#ffffff', map: fauxBrickTexture(), roughness: 0.85 }),
    blackPlastic: solid('#19191b', 0.5),
    greyPlastic: solid('#8d9296', 0.55),
    whitePlastic: solid('#efede8', 0.4),
    sack: solid('#e0cfa8', 1),
    paperboard: solid('#bdb3a2', 0.9),
    cardboard: solid('#b58e5f', 0.9),
    cableOrange: solid('#a2552b', 0.6),
    paper: solid('#f4f1ea', 0.8),
    solid,
  });
}

export type TroneMaterials = Awaited<ReturnType<typeof troneMaterials>>;
