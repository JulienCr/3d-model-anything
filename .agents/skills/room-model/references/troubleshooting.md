# Troubleshooting

Mirrors section 3 of `docs/room-model-guide.html` (French). Symptom, cause, remedy.

| Symptom | Cause | Remedy |
|---|---|---|
| Dimensions not visible in Blender | They are MeasureIt segments, hidden until shown | N sidebar, MeasureIt tab, **Show**. If MeasureIt is missing, `blender_report.json` says the native fallback ran; `ROOM_MODEL_NO_MEASUREIT=1` forces it. |
| `Blender not found: ... (set BLENDER)` | `BLENDER` empty or wrong; from WSL the Windows exe is under `/mnt/c/...` | `export BLENDER="/mnt/c/.../blender.exe"` (quoted). `wslpath` must exist. In `all` the stage is skipped with this message; alone it fails. |
| Blender ignores an environment variable | A Windows `.exe` only sees variables listed in `WSLENV` | The pipeline adds `ROOM_MODEL_NO_MEASUREIT` only; add others to `WSLENV` yourself. |
| `stage ... exists, skipped (use --force)` though something changed | Only certain files make a stage stale, by mtime; an old-dated copy triggers nothing; code changes never trigger | `--force` on the wanted stage; `touch` an edited input. |
| `vision/answers.json` only holds `error` | `claude -p` failed (timeout, exit code, unreadable answer); not cached | Rerun the stage without `--force` (an error answer is always redone); `rejected.json` keeps the reason. `{"skipped": true}` comes from `--no-llm`. `claude` not found: install it or set `ROOM_MODEL_CLAUDE`. |
| A window is not measured (candidates only) | Curtain, frosted glass or furniture defeats the mask, then the LLM sees no edge | Genuine occlusion: tape the leaf (badge leaf / both leaves) or use `overrides.json`. |
| Door wider than its hole, or `to_verify` | Photo and scan differ by more than 3 sigma, or the photo caught the casing line | A leaf larger than the hole is refused; tape the leaf, the tape wins. |
| Values vanished from the survey page | Browser storage key changed once (page title to capture id); another browser or profile has its own storage | The page offers the old values in a banner; otherwise link `measurements.json` or import it. Keep the file linked: it is the durable copy. |
| Furniture or filler patches in textures | An object within about 5-6 cm of the wall escapes the mask; texels no photo covers are filled with a tinted CC0 texture | Known limit. Filler means a coverage gap (`textures/textures.json`): more photos of the area. The first fill downloads a CC0 texture (network). |
| Walls not square | Out-of-square flag and recalibration are specified, not yet in the code | Read the orthogonality section of `report.md` and decide with the tape. |
| Walls offset or very short, `shifted_seed`, `low_support`, `roomplan_fallback` | Real surface far from RoomPlan, little LiDAR, or no fit | Read the per-wall table of `report.md`; sigma grows accordingly; a tape station corrects the wall with the larger sigma. |
| Beam with "?" on the plan or an element under `A verifier` | Probable beam whose two sides are not seen, or a duct pressed on the ceiling | Check on site; remove a wrong element with `"remove": true` if its kind allows. |
| CUDA unavailable, `anchor` very slow | `torch.cuda.is_available()` is False, SAM runs on CPU | Fix the driver or install a CUDA torch in the venv. |
| `anchor` fails offline on first run | `facebook/sam2.1-hiera-large` not cached | Connect once; the download is reused. |
| `no photos for room N` or capture refused | Room without photos, or folder without `manifest.json` (zip not extracted, wrong folder level) | In `all` photo stages are skipped. Point `--capture` at the folder holding `manifest.json`; check `--room`. |
| Blender console shows errors from other add-ons | Faulty personal add-ons log at each start | Harmless. The real failure is at the end of `blender.log` in `--out`. |

For anything else: `report.md` (alerts, flags), `blender.log`, and the tests in `pipelines/room_model/` to validate the install.
