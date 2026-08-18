# Local MediaPipe models

Place these files in this directory before starting the app:

- `hand_landmarker.task`
- `face_landmarker.task`

From the repository root, the setup helper downloads the official MediaPipe
float16 bundles once:

```powershell
python scripts\download_models.py
```

The application only reads these files locally after setup. Model binaries are
ignored by Git because they are large; the helper is safe to run again and will
not replace existing files.
