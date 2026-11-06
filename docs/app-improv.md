# House Capture app: improvements suggested by the room-model pipeline

The room-model pipeline (`docs/specs/room-model-pipeline.md`) turns a capture into dimensions. Runs on
real rooms showed where the capture limits the result. This list proposes changes to
`apps/house-capture`, most useful first. It states no figure from a real room.

## What the app already does

- `CoverageEngine` scores every LiDAR mesh point, and RoomPlan's own surfaces, by photo quality. The live overlay shows
  « needs photos / almost / done ».
- Auto-photo shoots when a view adds coverage (`engine.latestGain > minimumGain`), at most every 0.4 s.
  It also rejects blurred frames.
- The capture counts as ready at 3 walls, 8 photos and 85 % photo coverage (`CaptureQuality`).

All of this measures **photo** coverage. Nothing tells the user what the **LiDAR** has missed. The
pipeline's measurements, however, come from LiDAR.

## 1. Show what the LiDAR has not seen

**Problem.** A share of the ceiling had no LiDAR surface, and several short walls had very little LiDAR
support (short returns, the sides of a pillar). Their angles are then unreliable: a fitted corner can
read well away from the right angle it has in reality.

**Proposal.**
- A separate overlay layer for "no mesh here" on RoomPlan's surfaces. RoomPlan's walls and ceiling
  give a target that LiDAR holes can be measured against, which the engine already samples.
- A per-surface LiDAR gauge: ceiling %, and m² per wall with short walls highlighted.
- A readiness rule on LiDAR, not only on photos: ceiling ≥ 90 %, and every wall ≥ 0.5 m² or
  explicitly skipped.

## 2. Take more photos, on purpose

**Problem.** Auto-photo stops once coverage is reached, so each surface tends to be seen once. In
the pose refinement, a 3D point is seen by only a few photos on average, and some photos could not be
refined for lack of matches. Thin pipes (later milestones) need at least 3 views with real parallax.

**Proposal.**
- A "geometry" photo density setting: a target of N views per surface point (default 3), from
  directions at least 15° apart, instead of a single view.
- A manual burst mode for pipes, beams and door frames: walk along the element while the app
  shoots every 20–30 cm of displacement.
- A hard cap stays (storage), shown as a counter.

## 3. Guided ceiling pass

**Problem.** Ceilings carry what matters most after walls: levels, beams and their undersides, ducts.
People scan at eye height and tilt the phone up too little. A ceiling with several levels and a beam
is easy to under-scan, and the unseen share of the ceiling is where such holes sit.

**Proposal.** A short guided step, "point at the ceiling and walk the room". It ends when ceiling
LiDAR coverage reaches the target, with beams highlighted while they are being swept.

## 4. Warn on low-texture photos

**Problem.** Close-ups of blank walls cannot be matched, which is what keeps some photos at their
ARKit pose.

**Proposal.** Compute a cheap feature count (corners) on the preview. When it is low, prefer a frame
that includes an edge, a corner or an object, or ask the user to step back.

## 5. Capture wall thickness at openings

**Problem.** Wall thickness is assumed (20 cm) unless both faces or a jamb are scanned.

**Proposal.** When RoomPlan reports a door or an opening, prompt for a quick sweep of its reveal
(jamb depth). If the neighbouring room is captured in the same session, link the two faces.

## 6. A checklist before leaving the site

**Proposal.** Show one screen before closing the session, with:
- LiDAR coverage per wall and ceiling;
- photo coverage;
- weak photos;
- surfaces RoomPlan has not seen.

Each line is tappable to go back to that spot. A return visit costs far more than two extra minutes
on site.

## 7. Later: markers and on-site measurements

- Detect the printable AprilTag markers (spike `markers/`) and save their poses. This gives an independent
  scale check and helps merge rooms.
- The measurement page (milestone M4) is a web page with a shared database. If it moves into the app,
  a measurement could be typed right where its photo crop was taken.
