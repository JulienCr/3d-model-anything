// node --test tools/drive-sync.test.mjs   (fake drive and fake local root in a temp dir; needs zip and unzip)
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { runWatched } from './drive-sync.mjs';

const script = path.join(path.dirname(fileURLToPath(import.meta.url)), 'drive-sync.mjs');
const write = (file, content) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, content); };
const tmpdir = () => fs.mkdtempSync(path.join(os.tmpdir(), 'drive-sync-test-'));

/** A fake drive holding one room ("Trone") the way the real one is laid out. */
function fakeDrive(root) {
  const d = path.join(root, 'Trone');
  write(path.join(d, 'room.json'), '{"a":1}');
  write(path.join(d, 'web-assets/photos/0000.jpg'), 'jpg-bytes');
  write(path.join(d, 'scans/scan_room.ply'), 'ply-bytes-longer');
  const src = tmpdir();
  write(path.join(src, 'Room/manifest.json'), '{}');
  write(path.join(src, 'Room/mesh/room-0.ply'), 'mesh');
  fs.mkdirSync(path.join(d, 'capture'), { recursive: true });
  const r = spawnSync('zip', ['-qr', path.join(d, 'capture/Trone-iphone-export.zip'), 'Room'], { cwd: src });
  assert.equal(r.status, 0, 'zip CLI needed for this test');
  fs.rmSync(src, { recursive: true, force: true });
}

function run(args, env, home = os.tmpdir()) {
  return spawnSync(process.execPath, [script, ...args], { encoding: 'utf8', env: { ...process.env, HOME: home, ...env } });
}

describe('drive-sync', () => {
  for (const copier of ['rsync', '/nonexistent-copier']) {
    it(`pulls, is idempotent, never leaves staging files (${copier === 'rsync' ? 'rsync' : 'cp fallback'})`, () => {
      const root = tmpdir(), drive = path.join(root, 'drive'), local = path.join(root, 'local');
      fakeDrive(drive);
      const env = { DRIVE_DIR: drive, DRIVE_SYNC_LOCAL_ROOT: local, DRIVE_SYNC_COPY_BIN: copier };
      const first = run(['pull', 'trone'], env);
      assert.equal(first.status, 0, first.stderr + first.stdout);
      assert.equal(fs.readFileSync(path.join(local, 'data/demos/trone/room.json'), 'utf8'), '{"a":1}');
      assert.ok(fs.existsSync(path.join(local, 'data/web-assets/trone/photos/0000.jpg')));
      assert.ok(fs.existsSync(path.join(local, 'data/scans/trone/scan_room.ply')));
      assert.ok(fs.existsSync(path.join(local, 'data/captures-in/trone/mesh/room-0.ply')), 'top folder stripped');
      assert.ok(!fs.existsSync(path.join(local, 'data/captures-in/trone/Room')));
      const leftovers = spawnSync('find', [path.join(local, 'data'), '-name', '.drive-sync-*'], { encoding: 'utf8' }).stdout.trim();
      assert.equal(leftovers, '');
      const second = run(['pull', 'trone'], env);
      assert.match(second.stdout, /web-assets: pull 0 files/);
      assert.match(second.stdout, /capture: pull 0 files/);
      fs.rmSync(root, { recursive: true, force: true });
    });
  }

  it('push never overwrites a different drive file without --force, and never deletes', () => {
    const root = tmpdir(), drive = path.join(root, 'drive'), local = path.join(root, 'local');
    fakeDrive(drive);
    const env = { DRIVE_DIR: drive, DRIVE_SYNC_LOCAL_ROOT: local };
    assert.equal(run(['pull', 'trone'], env).status, 0);
    write(path.join(local, 'data/web-assets/trone/photos/0000.jpg'), 'changed-and-longer');
    write(path.join(local, 'data/web-assets/trone/photos/0001.jpg'), 'new');
    write(path.join(drive, 'Trone/web-assets/only-on-drive.txt'), 'keep');
    const refused = run(['push', 'trone'], env);
    assert.equal(refused.status, 2, refused.stdout + refused.stderr);
    assert.equal(fs.readFileSync(path.join(drive, 'Trone/web-assets/photos/0000.jpg'), 'utf8'), 'jpg-bytes');
    assert.equal(fs.readFileSync(path.join(drive, 'Trone/web-assets/photos/0001.jpg'), 'utf8'), 'new', 'missing files are still pushed');
    assert.equal(run(['push', 'trone', '--force'], env).status, 0);
    assert.equal(fs.readFileSync(path.join(drive, 'Trone/web-assets/photos/0000.jpg'), 'utf8'), 'changed-and-longer');
    assert.ok(fs.existsSync(path.join(drive, 'Trone/web-assets/only-on-drive.txt')));
    const dry = run(['push', 'trone', '--dry-run'], env);
    assert.match(dry.stdout, /dry run/);
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('pushes a capture zip for a room the drive does not have yet', () => {
    const root = tmpdir(), drive = path.join(root, 'drive'), local = path.join(root, 'local');
    fs.mkdirSync(drive, { recursive: true });
    write(path.join(local, 'data/demos/trone/room.json'), '{}');
    write(path.join(local, 'data/captures-in/trone/manifest.json'), '{}');
    write(path.join(local, 'data/captures-in/trone/mesh/room-0.ply'), 'mesh');
    const env = { DRIVE_DIR: drive, DRIVE_SYNC_LOCAL_ROOT: local };
    assert.equal(run(['push', 'trone'], env).status, 0);
    assert.ok(fs.existsSync(path.join(drive, 'Trone/capture/Trone-iphone-export.zip')));
    fs.rmSync(path.join(local, 'data/captures-in'), { recursive: true });
    assert.equal(run(['pull', 'trone'], env).status, 0);
    assert.ok(fs.existsSync(path.join(local, 'data/captures-in/trone/mesh/room-0.ply')));
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('aborts a stalled copy after one retry, naming the file and the offline hint, leaving nothing in place', () => {
    const root = tmpdir(), drive = path.join(root, 'drive'), local = path.join(root, 'local');
    fakeDrive(drive);
    const fake = path.join(root, 'fake-rsync.sh');
    write(fake, '#!/bin/sh\n[ "$1" = "--version" ] && exit 0\nexec sleep 60\n');
    fs.chmodSync(fake, 0o755);
    const r = run(['pull', 'trone'], { DRIVE_DIR: drive, DRIVE_SYNC_LOCAL_ROOT: local, DRIVE_SYNC_COPY_BIN: fake, DRIVE_SYNC_STALL_SECONDS: '1.5' });
    assert.equal(r.status, 1);
    assert.match(r.stdout, /retrying once/);
    assert.match(r.stderr, /no progress for 1\.5 s on .*room\.json/);
    assert.match(r.stderr, /Available offline/);
    assert.ok(!fs.existsSync(path.join(local, 'data/demos/trone/room.json')));
    assert.ok(!fs.existsSync(path.join(local, 'data/demos/trone/.drive-sync-stage')));
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('runWatched kills a silent process and lets a progressing one finish', async () => {
    const dir = tmpdir();
    const silent = await runWatched('sleep', ['30'], { watchDir: dir, label: 't', totalFiles: 1, totalSize: 1, stall: 800, poll: 100 });
    assert.equal(silent.stalled, true);
    const progressing = await runWatched('sh', ['-c', `for i in 1 2 3 4 5 6; do echo $i >> "${dir}/f$i"; sleep 0.3; done`], { watchDir: dir, label: 't', totalFiles: 6, totalSize: 12, stall: 800, poll: 100 });
    assert.deepEqual([progressing.stalled, progressing.status], [false, 0]);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('auto-detects a single Drive folder and refuses several, listing them', () => {
    const home = tmpdir();
    const make = account => {
      const d = path.join(home, 'Library/CloudStorage', `GoogleDrive-${account}`, 'Shared drives/Team/Salle/3D/3d-model-anything');
      fakeDrive(d);
      return d;
    };
    make('a@example.org');
    const local = path.join(home, 'local');
    const one = run(['status', 'trone'], { DRIVE_SYNC_LOCAL_ROOT: local, DRIVE_DIR: '' }, home);
    assert.equal(one.status, 0, one.stderr);
    assert.match(one.stdout, /drive: .*GoogleDrive-a@example\.org/);
    make('b@example.org');
    const two = run(['status', 'trone'], { DRIVE_SYNC_LOCAL_ROOT: local, DRIVE_DIR: '' }, home);
    assert.equal(two.status, 1);
    assert.match(two.stderr, /several Drive folders match[\s\S]*GoogleDrive-a@example\.org[\s\S]*GoogleDrive-b@example\.org/);
    fs.rmSync(home, { recursive: true, force: true });
  });
});
