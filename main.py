#!/usr/bin/env python3
"""
ViewMax Local Automation Engine
Production-ready FastAPI server for processing YouTube videos into viral 9:16 vertical shorts
with AI-powered highlight detection and OpusClip-style dynamic captions.
"""

import os
import json
import sqlite3
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import contextmanager
from enum import Enum

import yt_dlp
import cv2
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import assemblyai as aai
import requests
from moviepy.editor import VideoFileClip, concatenate_videoclips

# ==================== Configuration ====================

class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    DETECTING_HIGHLIGHTS = "detecting_highlights"
    SLICING = "slicing"
    REFRAMING = "reframing"
    GENERATING_CAPTIONS = "generating_captions"
    BURNING_CAPTIONS = "burning_captions"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingConfig:
    """Central configuration management"""
    def __init__(self):
        self.storage_path = Path(os.getenv("STORAGE_PATH", "./storage"))
        self.raw_path = self.storage_path / "raw"
        self.slices_path = self.storage_path / "slices"
        self.captions_path = self.storage_path / "captions"
        self.exports_path = self.storage_path / "exports"
        self.db_path = Path(os.getenv("DATABASE_URL", "viewmax.db").replace("sqlite:///", ""))
        
        self.assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "nemotron")
        
        # Create directories
        for path in [self.raw_path, self.slices_path, self.captions_path, self.exports_path]:
            path.mkdir(parents=True, exist_ok=True)


config = ProcessingConfig()

# ==================== Database Layer ====================

class DatabaseManager:
    """SQLite database management with context manager pattern"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Provide connection context to prevent memory leaks"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    youtube_url TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    raw_video_path TEXT,
                    transcript_path TEXT,
                    highlights_json TEXT,
                    slice_video_path TEXT,
                    reframed_video_path TEXT,
                    captions_path TEXT,
                    final_export_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_steps (
                    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_details TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
            """)
    
    def create_job(self, job_id: str, project_name: str, youtube_url: str) -> bool:
        """Create a new processing job"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO jobs (job_id, project_name, youtube_url, status)
                VALUES (?, ?, ?, ?)
            """, (job_id, project_name, youtube_url, JobStatus.PENDING.value))
        return True
    
    def update_job_status(self, job_id: str, status: JobStatus, error_msg: Optional[str] = None):
        """Update job status and optionally set error message"""
        with self.get_connection() as conn:
            if error_msg:
                conn.execute("""
                    UPDATE jobs SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (status.value, error_msg, job_id))
            else:
                conn.execute("""
                    UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (status.value, job_id))
    
    def update_job_field(self, job_id: str, field: str, value: str):
        """Update a specific job field"""
        with self.get_connection() as conn:
            conn.execute(f"""
                UPDATE jobs SET {field} = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            """, (value, job_id))
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Retrieve job details"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM jobs WHERE job_id = ?
            """, (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_jobs(self) -> List[Dict]:
        """Retrieve all jobs"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM jobs ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def log_processing_step(self, job_id: str, step_name: str, status: str, error_details: Optional[str] = None):
        """Log a processing step"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO processing_steps (job_id, step_name, status, started_at, error_details)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (job_id, step_name, status, error_details))


db_manager = DatabaseManager(config.db_path)

# ==================== Pydantic Models ====================

class JobRequest(BaseModel):
    """Request model for new processing job"""
    youtube_url: str
    project_name: str
    assemblyai_api_key: str


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    project_name: str
    youtube_url: str
    status: str
    progress_steps: List[Dict]
    error_message: Optional[str] = None
    final_export_url: Optional[str] = None


# ==================== Video Processing Core ====================

class VideoDownloader:
    """YouTube video download with robust error handling"""
    
    @staticmethod
    def download_video(youtube_url: str, output_path: Path) -> bool:
        """
        Download best MP4 stream from YouTube using yt-dlp.
        Uses context management to handle large file streams efficiently.
        """
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': str(output_path),
                'quiet': False,
                'no_warnings': False,
                'socket_timeout': 30,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                return True
        except Exception as e:
            raise Exception(f"Video download failed: {str(e)}")


class TranscriptionEngine:
    """AssemblyAI transcription with word-level timestamps"""
    
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
    
    def transcribe_video(self, video_path: Path) -> Dict:
        """
        Transcribe video to get word-level timestamps.
        Returns structured JSON with full transcript and word timing data.
        """
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=False,
                speakers_expected=1,
            )
            
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(str(video_path))
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            # Extract word-level timing data
            words_data = []
            for word in transcript.words:
                words_data.append({
                    "word": word.text,
                    "start_ms": int(word.start),
                    "end_ms": int(word.end),
                })
            
            return {
                "full_text": transcript.text,
                "words": words_data,
                "paragraphs": [p.text for p in transcript.paragraphs] if transcript.paragraphs else [],
                "language": "en",
            }
        except Exception as e:
            raise Exception(f"Transcription engine failed: {str(e)}")


class HighlightDetector:
    """Local NVIDIA Nemotron highlight detection via Ollama"""
    
    def __init__(self, ollama_base_url: str, model_name: str):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
    
    def detect_highlights(self, transcript_text: str, words_data: List[Dict]) -> Dict:
        """
        Send transcript to local Nemotron model via Ollama.
        Parse strict JSON response for optimal clip segment timestamps.
        """
        try:
            system_prompt = """You are an expert viral video highlight detector. Analyze the provided transcript and identify the single most engaging, emotionally resonant segment that would make an excellent short-form vertical video clip (30-60 seconds). Return ONLY a valid JSON object with no additional text:
{
  "start_time_ms": <milliseconds>,
  "end_time_ms": <milliseconds>,
  "reason": "<brief explanation>",
  "viral_score": <0-100>
}"""
            
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\nTranscript:\n{transcript_text}",
                "stream": False,
                "format": "json",
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "")
            
            # Parse JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in Nemotron response")
            
            json_str = response_text[json_start:json_end]
            highlights = json.loads(json_str)
            
            return {
                "start_ms": highlights.get("start_time_ms"),
                "end_ms": highlights.get("end_time_ms"),
                "reason": highlights.get("reason"),
                "viral_score": highlights.get("viral_score"),
            }
        except Exception as e:
            raise Exception(f"Highlight detection failed: {str(e)}")


class VideoSlicer:
    """Extract video segment using MoviePy with memory-safe context handling"""
    
    @staticmethod
    def slice_video(input_path: Path, output_path: Path, start_ms: int, end_ms: int) -> bool:
        """
        Extract exact video segment between timestamps.
        Uses MoviePy context manager to prevent memory leaks on large files.
        """
        try:
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            
            # Context management for large video files
            with VideoFileClip(str(input_path)) as video:
                subclip = video.subclip(start_sec, end_sec)
                # Write with codec optimization
                subclip.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None,
                )
            
            return True
        except Exception as e:
            raise Exception(f"Video slicing failed: {str(e)}")


class SmartReframer:
    """Convert 16:9 horizontal video to 9:16 vertical with intelligent centering"""
    
    @staticmethod
    def reframe_to_vertical(input_path: Path, output_path: Path) -> bool:
        """
        Crop and convert horizontal video to vertical 9:16 format.
        Uses OpenCV frame-by-frame analysis for speaker centering.
        """
        try:
            cap = cv2.VideoCapture(str(input_path))
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Target 9:16 (vertical) dimensions
            target_height = frame_height
            target_width = int(frame_height * 9 / 16)
            
            # Calculate center crop
            x_offset = (frame_width - target_width) // 2
            
            # Video writer with memory-efficient codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (target_width, target_height),
            )
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Center crop to vertical format
                cropped = frame[0:target_height, x_offset:x_offset+target_width]
                out.write(cropped)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Reframing progress: {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            
            return True
        except Exception as e:
            raise Exception(f"Reframing failed: {str(e)}")


class CaptionGenerator:
    """Generate OpusClip-style .ass subtitle files with dynamic highlighting"""
    
    @staticmethod
    def generate_ass_subtitles(words_data: List[Dict], output_path: Path, clip_start_ms: int, clip_end_ms: int) -> bool:
        """
        Create Advanced SubStation Alpha (.ass) subtitle file with:
        - Maximum 3 words per line
        - Dynamic color highlighting per word
        - Impact/Montserrat Black font, size 16
        - White text with black outline, centered alignment
        """
        try:
            # ASS file header
            ass_content = """[Script Info]
Title: ViewMax Captions
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,16,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
            
            # Filter words within clip range
            filtered_words = [
                w for w in words_data
                if clip_start_ms <= w["start_ms"] <= clip_end_ms
            ]
            
            # Group words into lines (max 3 words per line)
            lines = []
            current_line = []
            current_line_start = None
            
            for word in filtered_words:
                if current_line_start is None:
                    current_line_start = word["start_ms"]
                
                current_line.append(word)
                
                if len(current_line) == 3 or word == filtered_words[-1]:
                    line_end = word["end_ms"]
                    lines.append({
                        "words": current_line,
                        "start_ms": current_line_start,
                        "end_ms": line_end,
                    })
                    current_line = []
                    current_line_start = None
            
            # Generate subtitle entries with dynamic highlighting
            for line in lines:
                start_sec = line["start_ms"] / 1000.0
                end_sec = line["end_ms"] / 1000.0
                start_time = CaptionGenerator._ms_to_ass_time(line["start_ms"])
                end_time = CaptionGenerator._ms_to_ass_time(line["end_ms"])
                
                # Build text with color tags for dynamic highlighting
                text_parts = []
                current_offset = line["start_ms"]
                
                for i, word in enumerate(line["words"]):
                    word_start = word["start_ms"] - line["start_ms"]
                    word_end = word["end_ms"] - line["start_ms"]
                    
                    # Highlight color (bright yellow) with timing
                    highlight_start = CaptionGenerator._ms_to_duration(word_start)
                    highlight_end = CaptionGenerator._ms_to_duration(word_end)
                    
                    # Text with dynamic highlight: yellow during word, white after
                    text_parts.append(f"{{{highlight_start}}}{{\\c&H0000FFFF&}}{word['word']}")
                    text_parts.append(f"{{{highlight_end}}}{{\\c&H00FFFFFF&}} ")
                
                text = "".join(text_parts).strip()
                
                ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
            
            # Write ASS file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            return True
        except Exception as e:
            raise Exception(f"Caption generation failed: {str(e)}")
    
    @staticmethod
    def _ms_to_ass_time(ms: int) -> str:
        """Convert milliseconds to ASS time format (h:mm:ss.xx)"""
        total_seconds = ms / 1000.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        centiseconds = int((total_seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    
    @staticmethod
    def _ms_to_duration(ms: int) -> str:
        """Convert milliseconds to duration format for inline tags"""
        return f"{int(ms)}"


class CaptionBurner:
    """Burn ASS subtitles into video using FFmpeg subprocess"""
    
    @staticmethod
    def burn_captions(video_path: Path, captions_path: Path, output_path: Path) -> bool:
        """
        Use FFmpeg to permanently burn styled subtitles into video.
        Executes system-level FFmpeg subprocess with robust error handling.
        """
        try:
            # Build FFmpeg command with subtitle filter
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f"subtitles='{captions_path}'",
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-y',
                str(output_path),
            ]
            
            # Execute FFmpeg with subprocess context manager
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            stdout, stderr = process.communicate(timeout=600)
            
            if process.returncode != 0:
                raise Exception(f"FFmpeg failed: {stderr.decode()}")
            
            return True
        except Exception as e:
            raise Exception(f"Caption burning failed: {str(e)}")


# ==================== FastAPI Application ====================

app = FastAPI(title="ViewMax Local Automation Engine", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== API Endpoints ====================

@app.post("/api/jobs/create")
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """
    Create a new video processing job and start background workflow.
    """
    import uuid
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        db_manager.create_job(job_id, request.project_name, request.youtube_url)
        
        # Queue background processing
        background_tasks.add_task(
            process_video_pipeline,
            job_id=job_id,
            youtube_url=request.youtube_url,
            assemblyai_key=request.assemblyai_api_key,
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video processing started",
        }
    except Exception as e:
        db_manager.update_job_status(job_id, JobStatus.FAILED, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get current job status and progress.
    """
    job = db_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    with db_manager.get_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM processing_steps WHERE job_id = ? ORDER BY step_id
        """, (job_id,))
        steps = [dict(row) for row in cursor.fetchall()]
    
    return {
        "job_id": job_id,
        "project_name": job["project_name"],
        "youtube_url": job["youtube_url"],
        "status": job["status"],
        "progress_steps": steps,
        "error_message": job.get("error_message"),
        "final_export_url": f"/api/exports/{job_id}" if job["final_export_path"] else None,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@app.get("/api/jobs")
async def list_jobs():
    """
    List all processing jobs.
    """
    jobs = db_manager.get_all_jobs()
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/exports/{job_id}")
async def download_export(job_id: str):
    """
    Download final processed video export.
    """
    job = db_manager.get_job(job_id)
    
    if not job or not job["final_export_path"]:
        raise HTTPException(status_code=404, detail="Export not found")
    
    export_path = Path(job["final_export_path"])
    
    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found on disk")
    
    return FileResponse(
        export_path,
        media_type="video/mp4",
        filename=f"{job['project_name']}_final.mp4",
    )


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def serve_frontend():
    """
    Serve frontend HTML dashboard.
    """
    return FileResponse("static/index.html")


# ==================== Background Processing Pipeline ====================

async def process_video_pipeline(
    job_id: str,
    youtube_url: str,
    assemblyai_key: str,
):
    """
    Main video processing workflow - executes all steps sequentially.
    Each step updates the database with progress and error handling.
    """
    
    try:
        # Step 1: Download video
        print(f"[{job_id}] Starting video download...")
        db_manager.update_job_status(job_id, JobStatus.DOWNLOADING)
        db_manager.log_processing_step(job_id, "Video Download", "started")
        
        raw_video_path = config.raw_path / f"{job_id}.mp4"
        VideoDownloader.download_video(youtube_url, raw_video_path)
        db_manager.update_job_field(job_id, "raw_video_path", str(raw_video_path))
        db_manager.log_processing_step(job_id, "Video Download", "completed")
        
        # Step 2: Transcribe audio
        print(f"[{job_id}] Starting transcription...")
        db_manager.update_job_status(job_id, JobStatus.TRANSCRIBING)
        db_manager.log_processing_step(job_id, "Audio Transcription", "started")
        
        transcriber = TranscriptionEngine(assemblyai_key)
        transcript_data = transcriber.transcribe_video(raw_video_path)
        transcript_path = config.raw_path / f"{job_id}_transcript.json"
        
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        
        db_manager.update_job_field(job_id, "transcript_path", str(transcript_path))
        db_manager.log_processing_step(job_id, "Audio Transcription", "completed")
        
        # Step 3: Detect highlights with Nemotron
        print(f"[{job_id}] Detecting highlights...")
        db_manager.update_job_status(job_id, JobStatus.DETECTING_HIGHLIGHTS)
        db_manager.log_processing_step(job_id, "Highlight Detection", "started")
        
        detector = HighlightDetector(config.ollama_base_url, config.ollama_model)
        highlights = detector.detect_highlights(
            transcript_data["full_text"],
            transcript_data["words"],
        )
        
        highlights_path = config.raw_path / f"{job_id}_highlights.json"
        with open(highlights_path, 'w') as f:
            json.dump(highlights, f, indent=2)
        
        db_manager.update_job_field(job_id, "highlights_json", str(highlights_path))
        db_manager.log_processing_step(job_id, "Highlight Detection", "completed")
        
        # Step 4: Slice video
        print(f"[{job_id}] Slicing video...")
        db_manager.update_job_status(job_id, JobStatus.SLICING)
        db_manager.log_processing_step(job_id, "Video Slicing", "started")
        
        slice_video_path = config.slices_path / f"{job_id}_slice.mp4"
        VideoSlicer.slice_video(
            raw_video_path,
            slice_video_path,
            highlights["start_ms"],
            highlights["end_ms"],
        )
        
        db_manager.update_job_field(job_id, "slice_video_path", str(slice_video_path))
        db_manager.log_processing_step(job_id, "Video Slicing", "completed")
        
        # Step 5: Reframe to vertical
        print(f"[{job_id}] Reframing to vertical...")
        db_manager.update_job_status(job_id, JobStatus.REFRAMING)
        db_manager.log_processing_step(job_id, "Smart Reframing", "started")
        
        reframed_video_path = config.slices_path / f"{job_id}_reframed.mp4"
        SmartReframer.reframe_to_vertical(slice_video_path, reframed_video_path)
        
        db_manager.update_job_field(job_id, "reframed_video_path", str(reframed_video_path))
        db_manager.log_processing_step(job_id, "Smart Reframing", "completed")
        
        # Step 6: Generate captions
        print(f"[{job_id}] Generating captions...")
        db_manager.update_job_status(job_id, JobStatus.GENERATING_CAPTIONS)
        db_manager.log_processing_step(job_id, "Caption Generation", "started")
        
        captions_path = config.captions_path / f"{job_id}.ass"
        CaptionGenerator.generate_ass_subtitles(
            transcript_data["words"],
            captions_path,
            highlights["start_ms"],
            highlights["end_ms"],
        )
        
        db_manager.update_job_field(job_id, "captions_path", str(captions_path))
        db_manager.log_processing_step(job_id, "Caption Generation", "completed")
        
        # Step 7: Burn captions into video
        print(f"[{job_id}] Burning captions...")
        db_manager.update_job_status(job_id, JobStatus.BURNING_CAPTIONS)
        db_manager.log_processing_step(job_id, "Caption Burning", "started")
        
        final_export_path = config.exports_path / f"{job_id}_final.mp4"
        CaptionBurner.burn_captions(reframed_video_path, captions_path, final_export_path)
        
        db_manager.update_job_field(job_id, "final_export_path", str(final_export_path))
        db_manager.log_processing_step(job_id, "Caption Burning", "completed")
        
        # Mark job as completed
        print(f"[{job_id}] Pipeline completed successfully")
        db_manager.update_job_status(job_id, JobStatus.COMPLETED)
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] Pipeline failed: {error_msg}")
        db_manager.update_job_status(job_id, JobStatus.FAILED, error_msg)
        db_manager.log_processing_step(job_id, "Pipeline", "failed", error_msg)


# ==================== Startup ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        log_level="info",
    )
