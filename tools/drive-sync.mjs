#!/usr/bin/env node
// Restore or publish the private room data (never in git) from/to a shared drive folder.
//   node tools/drive-sync.mjs pull|push|status [room ...] [--dry-run] [--force] [--with-exports]
// Rooms are the ones with a "drive" field in tools/rooms.json. Drive root: $DRIVE_DIR, else auto-detected
// under ~/Library/CloudStorage/GoogleDrive-*/<Drive partagés|Shared drives>/*/Salle/3D/3d-model-anything.
// Nothing is ever deleted. A file that exists on the destination with a different size is kept unless --force.
// Environment: DRIVE_SYNC_STALL_SECONDS (default 90) aborts a copy that makes no byte progress for that long.
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { repoRoot, ToolError } from './lib/browser.mjs';

const localRoot = path.resolve(process.env.DRIVE_SYNC_LOCAL_ROOT ?? repoRoot); // test hook: where data/ lives
const registry = JSON.parse(fs.readFileSync(path.join(repoRoot, 'tools/rooms.json'), 'utf8')).rooms.filter(r => r.drive);
const IGNORED = new Set(['.DS_Store', 'desktop.ini', 'Thumbs.db']);
const IGNORED_PREFIX = ['.tmp.drive', '~$', '.~lock', '.drive-sync-'];
const STAGE = '.drive-sync-stage';
const SHARED_DRIVE_NAMES = ['Drive partagés', 'Shared drives'];
const DRIVE_SUBPATH = ['Salle', '3D', '3d-model-anything'];
const stallMs = () => Number(process.env.DRIVE_SYNC_STALL_SECONDS ?? 90) * 1000;
const fmt = n => (n < 1e6 ? `${(n / 1e3).toFixed(0)} kB` : n < 1e9 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e9).toFixed(2)} GB`);
const OFFLINE_HINT = 'Drive for desktop streams files on demand: right-click the shared folder > Offline access > Available offline, wait for the sync to finish, then rerun (already copied files are skipped).';

const dirsIn = dir => { try { return fs.readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => d.name).sort(); } catch { return []; } };

function findDriveRoot() {
  if (process.env.DRIVE_DIR) {
    const dir = path.resolve(process.env.DRIVE_DIR);
    if (!fs.existsSync(dir)) throw new ToolError(`DRIVE_DIR does not exist: ${dir}`);
    return fs.realpathSync(dir);
  }
  const base = path.join(os.homedir(), 'Library/CloudStorage');
  const found = [];
  for (const account of dirsIn(base).filter(n => n.startsWith('GoogleDrive-'))) {
    for (const shared of SHARED_DRIVE_NAMES) {
      for (const drive of dirsIn(path.join(base, account, shared))) {
        const candidate = path.join(base, account, shared, drive, ...DRIVE_SUBPATH);
        if (fs.existsSync(candidate)) found.push(candidate);
      }
    }
  }
  if (found.length === 1) return fs.realpathSync(found[0]);
  if (!found.length) throw new ToolError(`no Drive folder found under ${base}/GoogleDrive-*/<${SHARED_DRIVE_NAMES.join('|')}>/*/${DRIVE_SUBPATH.join('/')}. Install Drive for desktop, make the shared drive available, or set DRIVE_DIR.`);
  throw new ToolError(`several Drive folders match (the same shared drive may be mounted by two accounts, each with its own cache); set DRIVE_DIR to the one that is synced:\n  ${found.join('\n  ')}`);
}

const inside = (root, p) => { const rel = path.relative(root, p); return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel)); };
/** Real path of the deepest existing ancestor, so a symlink cannot redirect a write outside its root. */
function realOfExisting(p) {
  let cur = p;
  while (!fs.existsSync(cur)) cur = path.dirname(cur);
  return path.join(fs.realpathSync(cur), path.relative(cur, p));
}
const ignored = name => IGNORED.has(name) || IGNORED_PREFIX.some(p => name.startsWith(p));

/** Relative path -> size of every regular file under dir (symlinks are never followed). Empty map when absent. */
function listFiles(dir, keepAll = false) {
  const files = new Map();
  const walk = rel => {
    for (const e of fs.readdirSync(path.join(dir, rel), { withFileTypes: true })) {
      if (!keepAll && ignored(e.name)) continue;
      const r = path.join(rel, e.name);
      if (e.isDirectory()) walk(r);
      else if (e.isFile()) files.set(r, fs.statSync(path.join(dir, r)).size);
    }
  };
  if (fs.existsSync(dir) && fs.lstatSync(dir).isDirectory()) walk('');
  return files;
}

const totalBytes = files => [...files.values()].reduce((a, b) => a + b, 0);

/** What copying src onto destFiles would do: missing files are copied, same-size files skipped, different sizes are conflicts. */
function plan(src, destFiles, force) {
  const copy = [], conflicts = [];
  let same = 0;
  for (const [rel, size] of src) {
    if (!destFiles.has(rel)) copy.push(rel);
    else if (destFiles.get(rel) === size) same++;
    else (force ? copy : conflicts).push(rel);
  }
  return { copy, conflicts, same, bytes: copy.reduce((a, r) => a + src.get(r), 0) };
}

/**
 * Runs a child process and kills it when the watched directory shows no byte progress for `stall` ms.
 * Resolves { stalled, status, error? }. Prints progress (files, bytes) every `tick` ms.
 */
export function runWatched(cmd, args, { watchDir, label, totalFiles, totalSize, stall = stallMs(), tick = 5000, poll = 500 }) {
  return new Promise(resolve => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'ignore', 'inherit'] });
    let last = '', lastChange = Date.now(), lastPrint = Date.now(), stalled = false, done = false;
    const finish = result => { if (!done) { done = true; clearInterval(timer); resolve(result); } };
    const timer = setInterval(() => {
      const seen = listFiles(watchDir, true);
      const sig = `${seen.size}:${totalBytes(seen)}`;
      if (sig !== last) { last = sig; lastChange = Date.now(); }
      if (Date.now() - lastPrint >= tick) {
        lastPrint = Date.now();
        console.log(`    ${label}: ${seen.size}/${totalFiles} files, ${fmt(totalBytes(seen))}/${fmt(totalSize)}`);
      }
      if (!stalled && Date.now() - lastChange >= stall) {
        stalled = true;
        child.kill('SIGKILL');
        setTimeout(() => finish({ stalled: true, status: null }), 3000).unref(); // a process stuck in a Drive read may not die at once
      }
    }, poll);
    child.on('error', error => finish({ stalled: false, status: null, error }));
    child.on('close', status => finish({ stalled, status }));
  });
}

const copyBin = () => process.env.DRIVE_SYNC_COPY_BIN ?? 'rsync';
const rsyncAvailable = () => spawnSync(copyBin(), ['--version'], { stdio: 'ignore' }).status === 0;

function verify(src, destFiles, rels, name) {
  const bad = rels.filter(r => destFiles.get(r) !== src.get(r));
  if (bad.length) throw new ToolError(`${name}: ${bad.length} files missing or of wrong size after copy (first: ${bad[0]})`);
}

/**
 * Copies rels (sizes in src) from srcDir to destDir through <destDir>/.drive-sync-stage, then renames them into place
 * after a size check, so the final location never holds a partial file. A stalled copy is retried once.
 */
async function copyFiles(srcDir, destDir, rels, src, name) {
  if (!rels.length) return;
  const stage = path.join(destDir, STAGE);
  fs.mkdirSync(destDir, { recursive: true });
  fs.rmSync(stage, { recursive: true, force: true }); // left by an interrupted run
  fs.mkdirSync(stage);
  const total = rels.reduce((a, r) => a + src.get(r), 0);
  const useRsync = rsyncAvailable();
  const listDir = fs.mkdtempSync(path.join(os.tmpdir(), 'drive-sync-'));
  const listFile = path.join(listDir, 'files');
  fs.writeFileSync(listFile, rels.join('\n') + '\n');
  try {
    for (let attempt = 1; ; attempt++) {
      const staged = listFiles(stage, true);
      if (rels.every(rel => staged.get(rel) === src.get(rel))) break;
      const [cmd, args] = useRsync
        ? [copyBin(), ['-rt', `--files-from=${listFile}`, srcDir + path.sep, stage + path.sep]]
        : ['sh', ['-c', 'while IFS= read -r f; do mkdir -p "$2/$(dirname "$f")" && cp -f "$1/$f" "$2/$f" || exit 1; done < "$3"', 'copy', srcDir, stage, listFile]];
      const r = await runWatched(cmd, args, { watchDir: stage, label: name, totalFiles: rels.length, totalSize: total });
      if (r.error) throw new ToolError(`cannot run ${cmd}: ${r.error.message}`);
      const after = listFiles(stage, true);
      const stuck = rels.find(rel => after.get(rel) !== src.get(rel));
      if (!r.stalled && r.status !== 0) throw new ToolError(`${name}: ${cmd} failed (exit ${r.status}) near ${path.join(srcDir, stuck ?? '')}`);
      if (r.stalled && attempt >= 2) throw new ToolError(`${name}: no progress for ${stallMs() / 1000} s on ${path.join(srcDir, stuck ?? '')}. ${OFFLINE_HINT}`);
      if (r.stalled) console.log(`    ${name}: stalled on ${stuck}, retrying once`);
      else if (stuck && attempt >= 2) throw new ToolError(`${name}: ${stuck} is still incomplete after the copy`);
    }
    for (const rel of rels) {
      const target = path.join(destDir, rel);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.renameSync(path.join(stage, rel), target);
    }
  } finally {
    fs.rmSync(listDir, { recursive: true, force: true });
    fs.rmSync(stage, { recursive: true, force: true });
  }
  verify(src, listFiles(destDir), rels, name);
}

/** Lists the files an archive would extract, with the single top-level folder stripped when there is one. */
function zipEntries(zip) {
  const r = spawnSync('unzip', ['-l', zip], { encoding: 'utf8', maxBuffer: 1 << 28, timeout: 300_000 });
  if (r.error?.code === 'ENOENT') throw new ToolError('unzip not found: install it (it ships with macOS and most Linux distributions)');
  if (r.error || r.status !== 0) throw new ToolError(`cannot read ${zip}: ${r.error?.message ?? r.stderr.trim()}. ${OFFLINE_HINT}`);
  const entries = [];
  for (const line of r.stdout.split('\n')) {
    const m = line.match(/^\s*(\d+)\s+\S+\s+\S+\s+(.+)$/);
    if (m && !m[2].endsWith('/') && !ignored(path.basename(m[2]))) entries.push([m[2], Number(m[1])]);
  }
  const tops = new Set(entries.map(([n]) => n.split('/')[0]));
  const strip = tops.size === 1 && entries.every(([n]) => n.includes('/'));
  return new Map(entries.map(([n, size]) => [strip ? n.slice(n.indexOf('/') + 1) : n, size]));
}

function topFolderOf(dir) {
  const entries = fs.readdirSync(dir).filter(n => !ignored(n));
  const only = entries.length === 1 ? path.join(dir, entries[0]) : null;
  return only && fs.statSync(only).isDirectory() ? only : dir;
}

async function extractZip(zip, destDir, force, dryRun) {
  const src = zipEntries(zip);
  const p = plan(src, listFiles(destDir), force);
  if (dryRun || !p.copy.length) return { src, ...p };
  fs.mkdirSync(path.join(localRoot, 'data'), { recursive: true });
  const tmp = fs.mkdtempSync(path.join(localRoot, 'data', '.drive-sync-extract-'));
  try {
    const r = await runWatched('unzip', ['-q', '-o', zip, '-d', tmp], { watchDir: tmp, label: 'capture', totalFiles: src.size, totalSize: totalBytes(src) });
    if (r.stalled) throw new ToolError(`capture: no progress for ${stallMs() / 1000} s reading ${zip}. ${OFFLINE_HINT}`);
    if (r.error || r.status !== 0) throw new ToolError(`unzip failed (${r.error?.message ?? `exit ${r.status}`})`);
    const extracted = topFolderOf(tmp);
    if (fs.existsSync(destDir)) await copyFiles(extracted, destDir, p.copy, src, 'capture');
    else {
      fs.mkdirSync(path.dirname(destDir), { recursive: true });
      fs.renameSync(extracted, destDir);
    }
    verify(src, listFiles(destDir), p.copy, 'capture');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  return { src, ...p };
}

/** Per room, per item: { name, local, drive, kind } where kind is 'file', 'dir' or 'zip'. */
function items(room, driveRoot, withExports) {
  const d = path.join(driveRoot, room.drive);
  const data = path.join(localRoot, 'data');
  const list = [
    { name: 'room.json', kind: 'file', local: path.join(data, 'demos', room.id, 'room.json'), drive: path.join(d, 'room.json') },
    { name: 'web-assets', kind: 'dir', local: path.join(data, 'web-assets', room.id), drive: path.join(d, 'web-assets') },
    { name: 'scans', kind: 'dir', local: path.join(data, 'scans', room.id), drive: path.join(d, 'scans') },
    { name: 'capture', kind: 'zip', local: path.join(data, 'captures-in', room.id), drive: path.join(d, 'capture', `${room.drive}-iphone-export.zip`) },
  ];
  if (withExports) list.push({ name: 'exports', kind: 'dir', local: path.join(data, 'exports', room.id), drive: path.join(d, 'exports') });
  for (const i of list) {
    if (!inside(realOfExisting(data), realOfExisting(i.local)) || !inside(driveRoot, realOfExisting(i.drive))) throw new ToolError(`refusing path outside the two roots: ${i.name}`);
  }
  return list;
}

const fileMap = file => (fs.existsSync(file) && fs.statSync(file).isFile() ? new Map([[path.basename(file), fs.statSync(file).size]]) : new Map());

/** Files of one side of an item, as a Map (for a zip on the drive side: what its extraction would contain). */
function side(item, which) {
  const p = item[which];
  if (item.kind === 'file') return fileMap(p);
  if (item.kind === 'zip' && which === 'drive') return fs.existsSync(p) ? zipEntries(p) : new Map();
  return listFiles(p);
}

function stateOf(local, drive) {
  if (!local.size && !drive.size) return 'absent on both';
  if (!local.size) return 'drive only';
  if (!drive.size) return 'local only';
  const diff = [...drive].filter(([r, s]) => local.get(r) !== s).length + [...local.keys()].filter(r => !drive.has(r)).length;
  return diff ? `differs (${diff} files)` : 'match';
}

function status(room, driveRoot, withExports) {
  console.log(`${room.id}:`);
  for (const item of items(room, driveRoot, withExports)) {
    const local = side(item, 'local'), drive = side(item, 'drive');
    console.log(`  ${item.name.padEnd(11)} local ${String(local.size).padStart(5)} files ${fmt(totalBytes(local)).padStart(9)} | drive ${String(drive.size).padStart(5)} files ${fmt(totalBytes(drive)).padStart(9)} | ${stateOf(local, drive)}`);
  }
}

/** Push the extracted capture as a zip, only when the drive has none: archives are not byte-reproducible, so never replaced silently. */
async function pushCapture(item, opts) {
  const src = listFiles(item.local);
  const none = { src, copy: [], conflicts: [], same: 0, bytes: 0 };
  if (!src.size) return { ...none, note: 'no local capture' };
  if (fs.existsSync(item.drive) && !opts.force) return { ...none, same: src.size, note: 'a capture archive already exists on the drive (use --force to replace it)' };
  const result = { ...none, copy: ['(zip)'], bytes: totalBytes(src), note: 'zipped from the extracted capture' };
  if (opts.dryRun) return result;
  if (spawnSync('zip', ['-v'], { stdio: 'ignore' }).error) throw new ToolError('zip not found: install it to push a capture');
  const tmp = fs.mkdtempSync(path.join(localRoot, 'data', '.drive-sync-zip-'));
  try {
    const name = path.basename(item.drive);
    const r = spawnSync('zip', ['-qr0', path.join(tmp, name), path.basename(item.local)], { cwd: path.dirname(item.local), stdio: 'inherit' });
    if (r.status !== 0) throw new ToolError(`zip failed (exit ${r.status})`);
    verify(src, zipEntries(path.join(tmp, name)), [...src.keys()], 'capture archive');
    const size = fs.statSync(path.join(tmp, name)).size;
    await copyFiles(tmp, path.dirname(item.drive), [name], new Map([[name, size]]), 'capture');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  return result;
}

async function transfer(direction, room, driveRoot, opts) {
  const pull = direction === 'pull';
  const summary = { copied: 0, bytes: 0, conflicts: 0, skipped: 0 };
  console.log(`${room.id}:`);
  for (const item of items(room, driveRoot, opts.withExports)) {
    const [from, to] = pull ? ['drive', 'local'] : ['local', 'drive'];
    const verb = `${opts.dryRun ? 'would ' : ''}${pull ? 'pull' : 'push'}`;
    if (!fs.existsSync(item[from])) { console.log(`  ${item.name}: nothing on ${from}, skipped`); continue; }
    let result;
    if (item.kind === 'zip') {
      result = pull ? await extractZip(item.drive, item.local, opts.force, opts.dryRun) : await pushCapture(item, opts);
    } else {
      const src = side(item, from);
      const p = plan(src, side(item, to), opts.force);
      result = { src, ...p };
      if (!opts.dryRun) {
        const dir = item.kind === 'file' ? path.dirname : x => x;
        await copyFiles(dir(item[from]), dir(item[to]), p.copy, src, `${room.id}/${item.name}`);
      }
    }
    summary.copied += result.copy.length; summary.bytes += result.bytes; summary.conflicts += result.conflicts.length; summary.skipped += result.same;
    const note = result.note ? `, ${result.note}` : '';
    console.log(`  ${item.name}: ${verb} ${result.copy.length} files (${fmt(result.bytes)}), ${result.same} already match${result.conflicts.length ? `, ${result.conflicts.length} DIFFER and are kept (use --force)` : ''}${note}`);
    for (const rel of result.conflicts.slice(0, 5)) console.log(`      differs: ${rel}`);
  }
  return summary;
}

const usage = () => 'usage: node tools/drive-sync.mjs pull|push|status [room ...] [--dry-run] [--force] [--with-exports]';

function parseArgs(argv) {
  const [command, ...rest] = argv;
  if (!['pull', 'push', 'status'].includes(command)) throw new ToolError(usage());
  const opts = { command, rooms: [], dryRun: false, force: false, withExports: false };
  for (const a of rest) {
    if (a === '--dry-run') opts.dryRun = true;
    else if (a === '--force') opts.force = true;
    else if (a === '--with-exports') opts.withExports = true;
    else if (a.startsWith('-')) throw new ToolError(`unknown option ${a}\n${usage()}`);
    else opts.rooms.push(a);
  }
  for (const r of registry) if (path.basename(r.drive) !== r.drive) throw new ToolError(`invalid "drive" for ${r.id} in tools/rooms.json`);
  const known = registry.map(r => r.id);
  for (const id of opts.rooms) if (!known.includes(id)) throw new ToolError(`unknown room "${id}" (rooms with a "drive" field in tools/rooms.json: ${known.join(', ')})`);
  if (!opts.rooms.length) opts.rooms = known;
  return opts;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const driveRoot = findDriveRoot();
  console.log(`drive: ${driveRoot}${opts.dryRun ? '  (dry run, nothing is written)' : ''}`);
  let conflicts = 0;
  for (const id of opts.rooms) {
    const room = registry.find(r => r.id === id);
    if (opts.command === 'status') status(room, driveRoot, opts.withExports);
    else conflicts += (await transfer(opts.command, room, driveRoot, opts)).conflicts;
  }
  if (conflicts) {
    console.error(`${conflicts} files differ between local and drive and were left untouched; rerun with --force to overwrite the destination.`);
    process.exit(2);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then(() => process.exit(0), error => {
    console.error(error instanceof ToolError ? `error: ${error.message}` : error);
    process.exit(1);
  });
}
