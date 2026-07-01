# ViewMax - Local AI-Powered Video Automation Engine

## 🎬 Overview

ViewMax is a **production-ready, local-first FastAPI application** that transforms long-form YouTube videos into viral 9:16 vertical shorts featuring **OpusClip-style dynamic animated captions**.

Built for a **single local PC** with zero Docker/cloud infrastructure overhead.

---

## 🚀 Architecture

### Tech Stack
- **Frontend:** HTML5/JavaScript + Tailwind CSS (Single-page dashboard)
- **Backend:** Python FastAPI on port 8000 with CORS enabled
- **Task Management:** FastAPI native `BackgroundTasks` (no Redis/Celery)
- **Database:** SQLite (`viewmax.db`) for job state tracking
- **Storage:** Local directory structure (`storage/raw`, `storage/slices`, `storage/exports`)
- **Video Processing:** MoviePy + FFmpeg (subtitles burning)
- **AI Models:**
  - **Transcription:** AssemblyAI SDK (word-level timestamps)
  - **Highlight Detection:** NVIDIA Nemotron via Ollama (localhost:11434)
  - **Visual Processing:** OpenCV (frame analysis, centering)
  - **YouTube Download:** yt-dlp

---

## 📋 Processing Pipeline

The automation follows a **linear 7-step workflow:**

```
1. VIDEO INGESTION & DOWNLOAD
   └─ yt-dlp downloads best MP4 from YouTube → storage/raw/{job_id}.mp4

2. AUDIO TRANSCRIPTION
   └─ AssemblyAI SDK transcribes → JSON with word-level timestamps

3. LOCAL NEMOTRON HIGHLIGHT DETECTION
   └─ Sends transcript to localhost Ollama (nemotron model)
   └─ Returns optimized start/end timestamps (milliseconds) for viral segment

4. VIDEO SLICER
   └─ MoviePy extracts exact segment → storage/slices/{job_id}_slice.mp4

5. SMART REFRAMER
   └─ OpenCV crops 16:9 → 9:16 vertical format, keeps speaker centered
   └─ Output: storage/slices/{job_id}_reframed.mp4

6. OPUSCLIP CAPTION ENGINE
   └─ Parses word timestamps, groups max 3 words per line
   └─ Generates .ass subtitle file with:
     • Impact/Montserrat Black font, 16px, white text, black outline
     • Dynamic color highlighting: Yellow flash for active words
     • Perfect center alignment (Alignment=5)

7. EXPORT
   └─ FFmpeg subprocess burns captions into video
   └─ Final output: storage/exports/{job_id}_final.mp4
```

---

## 🛠️ Installation

### Prerequisites
- **Python 3.9+**
- **FFmpeg** (system-level for caption burning)
- **Ollama** running locally on `http://localhost:11434` with `nemotron` model loaded
- **AssemblyAI API Key** (get from https://assemblyai.com)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set:
```
ASSEMBLYAI_API_KEY=your_actual_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nemotron
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
STORAGE_PATH=./storage
```

### Step 3: Ensure Ollama is Running

```bash
# In a separate terminal, ensure Ollama is running with nemotron model
oollama serve
# Then in another terminal:
ollama pull nemotron  # One-time download
```

### Step 4: Start FastAPI Server

```bash
python main.py
```

Server runs at `http://localhost:8000`

---

## 💻 Usage

### Web Dashboard

1. **Open browser:** `http://localhost:8000`

2. **Create new project:**
   - Project Name: e.g., "My First Short"
   - YouTube URL: e.g., `https://youtube.com/watch?v=...`
   - AssemblyAI API Key: Your actual key
   - Click "🚀 Start Processing"

3. **Monitor progress:**
   - Real-time step-by-step tracking
   - Status badges update every 2 seconds
   - Color-coded: pending → processing → completed/failed

4. **Download final video:**
   - Once "✅ Completed", click "⬇️ Download Final Video"
   - Or stream preview directly in browser

---

## 📡 API Endpoints

### Create Job
```http
POST /api/jobs/create
Content-Type: application/json

{
  "youtube_url": "https://youtube.com/watch?v=...",
  "project_name": "My Project",
  "assemblyai_api_key": "your_key"
}

Response:
{
  "job_id": "abc12345",
  "status": "queued",
  "message": "Video processing started"
}
```

### Get Job Status
```http
GET /api/jobs/{job_id}/status

Response:
{
  "job_id": "abc12345",
  "project_name": "My Project",
  "youtube_url": "...",
  "status": "completed",
  "progress_steps": [
    {"step_name": "Video Download", "status": "completed"},
    {"step_name": "Audio Transcription", "status": "completed"},
    ...
  ],
  "error_message": null,
  "final_export_url": "/api/exports/abc12345",
  "created_at": "2026-07-01T10:00:00",
  "updated_at": "2026-07-01T10:15:00"
}
```

### List All Jobs
```http
GET /api/jobs

Response:
{
  "jobs": [...],
  "total": 5
}
```

### Download Export
```http
GET /api/exports/{job_id}

Returns: MP4 video file
```

### Health Check
```http
GET /api/health

Response:
{
  "status": "healthy",
  "timestamp": "2026-07-01T10:00:00"
}
```

---

## 📁 Directory Structure

```
viewmax-local/
├── main.py                 # FastAPI server + processing pipeline
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── README.md              # This file
├── static/
│   └── index.html         # Single-page dashboard UI
├── storage/               # Auto-created
│   ├── raw/               # Downloaded YouTube videos
│   ├── slices/            # Extracted segments + reframed clips
│   ├── captions/          # .ass subtitle files
│   └── exports/           # Final processed videos with captions
└── viewmax.db             # SQLite database (auto-created)
```

---

## 🔧 Configuration

### Subtitle Styling

Edit `CaptionGenerator.generate_ass_subtitles()` in `main.py` to customize:

```python
# FontName options: 'Impact', 'Montserrat Black', 'Arial Black', etc.
# FontSize: default 16px
# PrimaryColour: White (&H00FFFFFF)
# OutlineColour: Black (&H00000000)
# Outline: 2px border
# Alignment: 5 = center
```

### Video Reframing

Adjust vertical crop logic in `SmartReframer.reframe_to_vertical()`:
- Modify `target_width = int(frame_height * 9 / 16)` for different aspect ratios
- Adjust `x_offset` calculation for left/right centering preferences

### FFmpeg Encoding

Modify `CaptionBurner.burn_captions()` FFmpeg command:
```python
ffmpeg_cmd = [
    'ffmpeg',
    '-i', str(video_path),
    '-vf', f"subtitles='{captions_path}'",
    '-c:v', 'libx264',
    '-preset', 'medium',  # fast/medium/slow for speed vs quality
    '-crf', '23',          # 18-28 (lower = better quality)
    '-c:a', 'aac',
    '-y',
    str(output_path),
]
```

---

## 🐛 Troubleshooting

### "FFmpeg not found"
```bash
# Install FFmpeg
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows:
choco install ffmpeg
```

### "Ollama connection refused"
```bash
# Ensure Ollama is running:
oollama serve
# In another terminal, pull nemotron:
ollama pull nemotron
```

### "AssemblyAI API error"
- Verify API key in `.env` is correct
- Check internet connection
- Ensure video duration is supported (typically < 3 hours)

### "Insufficient disk space"
- Raw videos can be 500MB+
- Each job needs: raw + slice + reframed + export storage
- Clean up `storage/` directory periodically

---

## 📊 Performance Notes

- **Video Download:** Depends on internet speed (typically 1-10 minutes for 1 hour video)
- **Transcription:** ~1-2 minutes per 10 minutes of video
- **Highlight Detection:** ~10-30 seconds (local Nemotron inference)
- **Video Processing:** ~5-15 minutes for 1 hour source → 45s clip (depends on CPU)
- **Total Time:** ~10-30 minutes for full pipeline on typical PC

---

## 📝 License

MIT License - Build amazing short-form videos! 🚀

---

## 🙋 Support

For issues or feature requests, open a GitHub issue with:
1. Error message/logs
2. Steps to reproduce
3. System specs (OS, Python version, CPU)

Happy clipping! 🎬✨
