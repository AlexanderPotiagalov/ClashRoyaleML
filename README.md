# KataCR-Lite: Milestone 0

This starter project does one thing:

> Read a Clash Royale replay video and export timestamped frames plus a JSONL manifest.

Do not install the full KataCR stack yet. First prove that your machine can read and process a replay reliably.

## 1. Requirements

- Windows 10/11
- Python 3.11
- Git is optional for this first milestone
- One Clash Royale replay saved as an MP4

## 2. Create the environment

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 and make sure "Add Python to PATH" is checked.

## 3. Check the installation

```powershell
python scripts/check_setup.py
```

Expected result:

```text
Python: OK
OpenCV: OK
NumPy: OK
Setup passed.
```

## 4. Add a replay

Put an MP4 inside:

```text
data/raw_videos/
```

Example:

```text
data/raw_videos/match_001.mp4
```

## 5. Inspect the replay

```powershell
python scripts/inspect_video.py data/raw_videos/match_001.mp4
```

This prints the resolution, FPS, duration, and frame count. It also saves the first frame to:

```text
data/frames/inspection/first_frame.jpg
```

## 6. Extract training frames

Start with two frames per second:

```powershell
python scripts/extract_frames.py data/raw_videos/match_001.mp4 --output data/frames/match_001 --sample-fps 2
```

Outputs:

```text
data/frames/match_001/
├── frame_000000_t000000ms.jpg
├── frame_000001_t000500ms.jpg
├── ...
└── manifest.jsonl
```

Each manifest row contains the frame path, source timestamp, video resolution, and crop configuration.

## Optional crop

The crop arguments are fractions of the original frame:

```powershell
python scripts/extract_frames.py data/raw_videos/match_001.mp4 `
  --output data/frames/match_001_arena `
  --sample-fps 2 `
  --crop 0.0 0.10 1.0 0.82
```

That example keeps the full width and crops approximately from 10% to 82% of the image height. Adjust it after viewing `first_frame.jpg`.

## Definition of done

Milestone 0 is complete when:

1. `check_setup.py` passes.
2. `inspect_video.py` reports believable video details.
3. At least 100 frames are exported.
4. The exported JPG files visibly show the match.
5. `manifest.jsonl` contains one row per exported frame.

Only then move to Milestone 1: arena cropping and object detection.
