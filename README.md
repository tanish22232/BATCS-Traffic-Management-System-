# traffic-violation-detection

## Structured deployment

### Run locally (Windows)
1. Double-click or run `scripts\run_api.bat` in a terminal. This will create a venv, install requirements, and start the API at `http://127.0.0.1:8000`.
2. Test upload:
   - Open `http://127.0.0.1:8000/docs` in a browser
   - Use the `POST /detect` endpoint to upload a video file
   - Response returns paths to the processed video under `output/` and CSV under `outputs/`

### Frontend (React)
1. In a new terminal, run the web app:
   ```
   scripts\run_web.bat
   ```
   - By default it points to `http://127.0.0.1:8000`. To target a different API port:
     ```
     set VITE_API_URL=http://127.0.0.1:8080
     scripts\run_web.bat
     ```
2. Open the shown local URL (e.g. `http://127.0.0.1:5173`) and upload a video.

### Run with Docker
```bash
docker build -t redlight-api .
docker run --rm -p 8000:8000 -v %cd%/output:/app/output -v %cd%/outputs:/app/outputs redlight-api
```
Then open `http://127.0.0.1:8000/docs` and use `POST /detect`.

### API
- `GET /health`: Service health
- `POST /detect`: Multipart upload `file` (video), returns JSON with `output_video` and `detections_csv` paths

