# Patent Demo Recording Guide

Goal: a short video of the 3D-printed chassis (Jetson Nano + camera + LCD)
that shows, on the device itself, a person's walk being captured, turned
into a gait skeleton, and stored as an enrolled identity.

Target length: **2-3 minutes**, one continuous take plus a few close-ups.
Everything is run on the Nano with a keyboard attached. No laptop in frame.

---

## What the audience sees (the story)

| # | Scene | On the LCD | What it proves |
|---|---|---|---|
| 1 | Chassis intro | Terminal, idle | The hardware: Nano, camera, LCD in one enclosure |
| 2 | Camera check | Raw camera view | The camera feeds the embedded board |
| 3 | Enrollment walk | Live camera + green skeleton overlay, `Pass N/5` | Pose is extracted from live video, on-device |
| 4 | Enrolled result | Black screen, skeleton replaying, `ENROLLED: <name>` | The walk became a stored gait signature |
| 5 | Proof it was saved | Terminal showing the database row | The signature persists in local storage |

Scenes 2 and 5 are short. Scene 3 is the core of the demo.

---

## A. Before you record (do this once, off camera)

Run these from the repo folder on the Nano:

```bash
cd ~/advanced-gait-analysis-on-embedded-hardware
source .venv/bin/activate
export DISPLAY=:0                       # needed if you ever use SSH; harmless otherwise
```

1. **Camera works and sees your full body.**
   ```bash
   python3 scripts/test_camera.py      # press q to close
   ```
   Stand where you will walk. Head to feet must be in frame.
2. **Clear old test enrollments** so the demo database is clean. List them, then delete each:
   ```bash
   sqlite3 database/gait.db "select person_name from enrolled_identities;"
   python3 -c "from src.database.gait_database import GaitDatabase; GaitDatabase().delete_identity('OLD_NAME')"
   ```
3. **Dry run the whole demo once** (Section C). The first run after boot is slow because
   torch and MediaPipe load. Do it once so the file cache is warm.
4. **Set the stage**
   - Plain background, even light, no one else walking through the frame.
   - Wear fitted, contrasting clothes. Baggy clothes and a cluttered background hurt pose tracking.
   - Mark the floor with tape: walk **across** the camera view, about 2-3 m away.
   - Mark the start and end points so each pass is the same.
   - Power the Nano from the proper 5V 4A supply. Run `sudo nvpmodel -m 0 && sudo jetson_clocks` for best speed.
   - Make sure the terminal font is large. The LCD is small and phone-sized video will shrink it further.
5. **Camera for the recording itself** (phone or a second camera). Frame so the chassis, the LCD
   text, and the Nano's camera lens are all visible. Do a test clip and check that the LCD text is readable.

---

## B. Shot list and voice-over

Read the voice-over lines at your own pace. Pause during the walk, since the on-screen text is the
content. These lines only describe what the screen actually shows. Do not quote accuracy numbers:
the model has not been validated on a large population, and the patent demo only needs to show the
working method.

### Scene 1 - Chassis (15 s)
Slow pan across the chassis: camera lens, LCD, Nano inside the enclosure.

> "This is the prototype: a compact enclosure housing an NVIDIA Jetson Nano, a camera, and a display.
> All processing happens on this board. Nothing is sent to a server."

### Scene 2 - Camera check (15 s)
```bash
python3 scripts/test_camera.py
```
The raw camera feed appears on the LCD. Wave once. Press `q`.

> "First, the camera feed. This is the live view that the system analyses."

### Scene 3 - Enrollment (60-90 s)
Type the command, then step back to the start mark while the model loads.

```bash
python3 scripts/enroll.py --name "Abhishek" --fullscreen
```

(Replace the name with whatever you want shown on screen.)

The LCD switches to the camera view. Walk across the frame at a normal pace, turn around, walk back.
Repeat until `Pass 5/5` is reached. After each pass a green `Pass N captured` flashes.

> "To enrol, the person simply walks across the camera's view. The system extracts a body skeleton
> from each video frame in real time. The green overlay is that skeleton: shoulders, hips, knees,
> ankles and feet. The green border means the body is being tracked."
>
> *(after the first pass captured)* "Each pass is checked for quality. A pass where the body was not
> clearly visible is rejected, and the person walks again."
>
> *(during later passes)* "From each pass the system builds a sequence of joint positions, joint
> angles and joint velocities. That's the gait data."

Enrollment ends by itself after 5 accepted passes. Do not press `q` unless you want to cancel:
`q` during the walk discards everything.

### Scene 4 - Enrolled gait skeleton (20 s)
The screen turns black and the captured skeleton plays back in a slow loop with
`ENROLLED: Abhishek` at the bottom. Let it loop two or three times.

> "Enrollment is complete. This is the stored gait skeleton from the best captured pass, replayed
> in slow motion. The averaged gait signature from all passes is now saved to the device's local database."

Press `q` to close the window.

### Scene 5 - Proof of storage (15 s)
Back at the terminal:

```bash
sqlite3 database/gait.db "select person_name, num_samples, quality_score from enrolled_identities;"
```

> "And here is the enrolled identity in the on-device database: the name, the number of walk samples
> used, and the pose-quality score."

Close with a shot of the whole chassis.

---

## C. Command cheat sheet (in order)

```bash
cd ~/advanced-gait-analysis-on-embedded-hardware && source .venv/bin/activate

python3 scripts/test_camera.py                                   # Scene 2 (q to close)
python3 scripts/enroll.py --name "Abhishek" --fullscreen         # Scenes 3 + 4 (q to close replay)
sqlite3 database/gait.db "select person_name, num_samples, quality_score from enrolled_identities;"   # Scene 5
```

Flags on `enroll.py`:

| Flag | Effect |
|---|---|
| `--fullscreen` | Preview and replay fill the LCD instead of a small window |
| `--no-replay` | Skip the enrolled-skeleton screen (not what you want for the demo) |
| `--no-display` | Terminal only, no windows (don't use for the demo) |

---

## D. If something goes wrong mid-take

| Symptom | Fix |
|---|---|
| Red border, "NO POSE - get full body in view" | Step back until head and feet are both in frame, improve lighting |
| "Pass rejected (xx%) - walk again" | Pose was lost for too many frames. Walk a bit slower, check clothing and background. It is a normal part of the flow and you can leave it in the video |
| Enrollment finishes with "Only captured N usable sequence(s)" | Fewer than 3 good passes. Re-run the same command |
| `q` pressed by mistake during the walk | Nothing is saved. Re-run the command |
| Window is tiny / not full screen | Add `--fullscreen`; check that the command was run from a terminal on the Nano's own display |
| `cv2.imshow` / display error | `export DISPLAY=:0` and run from the desktop terminal, not over SSH |
| Name already enrolled, or you want a clean slate | `python3 -c "from src.database.gait_database import GaitDatabase; GaitDatabase().delete_identity('Abhishek')"` |
| Process `Killed` | Out of RAM. Close the browser, confirm swap with `free -h` |

Record the walk scene in one take if you can. If it fails, just re-run. The deletion command above
resets the name for another try.

---

## E. After recording

- Trim to the scenes in the table at the top. Keep the walk and the enrolled-skeleton screen uncut.
- Add the voice-over afterwards over the footage (easier than speaking while walking). Use the
  lines in Section B as the script.
- Keep the raw unedited footage. The patent filing may want it.
- Also save a screenshot of the enrolled-skeleton screen and of the database row as still evidence.
