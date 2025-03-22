# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, AsyncIterator
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urljoin

import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .inference import generate_speech_from_api, AVAILABLE_VOICES, DEFAULT_VOICE


OUTPUTS_PATH = Path(__name__).parent.parent.parent.absolute() / 'outputs'
STATIC_PATH = Path(__name__).parent.parent.parent.absolute() / 'static'


class Config:
    def __init__(self, host: str, port: int, api_url: str, templates_dir: Path):
        self.host = host
        self.port = port
        self.api_url = api_url
        self.templates = Jinja2Templates(directory=templates_dir)
 
    def apply_to(self, func):
        def wrapper(*args, **kwargs):
            return func(self, *args, **kwargs)

        return wrapper

    @classmethod
    def from_args(cls, args) -> 'Config':
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--host", "-H", default="127.0.0.1", help="Hostname (or IP) to provide OpenAI-compatible TTS services on")
        arg_parser.add_argument("--port", "-P", default="5005", type=int, help="Port to provide OpenAI-compatible TTS services on")
        arg_parser.add_argument("--api-url", "-A", default="http://localhost:1337", help="URL of the inference server running the Orpheus model")

        parsed_args = arg_parser.parse_args(args)

        return cls(
            host=parsed_args.host,
            port=parsed_args.port,
            api_url=parsed_args.api_url,
            templates_dir=(Path(__file__).parent.parent.parent / 'templates').absolute(),
        )


class ThisApp(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = Config.from_args(sys.argv[1:])


# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

class APIResponse(BaseModel):
    status: str
    voice: str
    output_file: str
    generation_time: float


# Create FastAPI app
app = ThisApp(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0",
)


# OpenAI-compatible API endpoint
@app.post("/v1/audio/speech")
async def create_speech_api(app_request: Request, speech_request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    """
    if not speech_request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_PATH / f"{speech_request.voice}_{timestamp}.wav"
   
    output_url = urljoin('/outputs/', output_path.name)

    # Generate speech
    start = time.time()
    generate_speech_from_api(
        prompt=speech_request.input,
        voice=speech_request.voice,
        output_file=str(output_url),
        api_url=app_request.app.config.api_url,
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    # Return audio file
    return FileResponse(
        path=str(output_path),
        media_type="audio/wav",
        filename=f"{speech_request.voice}_{timestamp}.wav"
    )

# Legacy API endpoint for compatibility
@app.post("/speak")
async def speak(request: Request):
    """Legacy endpoint for compatibility with existing clients"""
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)

    if not text:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing 'text'"}
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_PATH / f"{voice}_{timestamp}.wav"
    
    # Generate speech
    start = time.time()
    generate_speech_from_api(prompt=text, voice=voice, output_file=str(output_path), api_url=request.app.config.api_url)
    end = time.time()
    generation_time = round(end - start, 2)
    
    output_url = urljoin('/outputs/', output_path.name)

    return JSONResponse(content={
        "status": "ok",
        "voice": voice,
        "output_file": str(output_url),
        "generation_time": generation_time
    })

# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to web UI"""
    return request.app.config.templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES}
    )

@app.get("/web/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Main web UI for TTS generation"""
    return request.app.config.templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES}
    )

@app.post("/web/", response_class=HTMLResponse)
async def generate_from_web(
    request: Request,
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE)
):
    """Handle form submission from web UI"""
    if not text:
        return request.app.config.templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": "Please enter some text.",
                "voices": AVAILABLE_VOICES
            }
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_PATH / f"{voice}_{timestamp}.wav"
    
    # Generate speech
    start = time.time()
    generate_speech_from_api(prompt=text, voice=voice, output_file=output_path, api_url=request.app.config.api_url)
    end = time.time()
    generation_time = round(end - start, 2)

    output_url = urljoin('/outputs/', output_path.name)

    return request.app.config.templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "success": True,
            "text": text,
            "voice": voice,
            "output_file": str(output_url),
            "generation_time": generation_time,
            "voices": AVAILABLE_VOICES
        }
    )


def path_is_within(input_path: Path, allowed_path: Path) -> bool:
    """
    Confirm that input_path is within allowed_path in absolute terms after resolving symlinks in allowed_path.
    
    Args:
        allowed_path (Path): The path within which input_path must be contained, after symlink resolution.
        input_path (Path): The path to check, absolute or relative.
    
    Returns:
        bool: True if the resolved input_path is within the resolved allowed_path, False otherwise.
    """
    # Convert input_path to absolute if it's relative, then resolve symlinks
    resolved_input_path = input_path.absolute().resolve()
    # Resolve allowed_path to its real, absolute location, following symlinks
    resolved_allowed_path = allowed_path.resolve()
    
    # Check if resolved_input_path is a subpath of resolved_allowed_path
    try:
        resolved_input_path.relative_to(resolved_allowed_path)
        return True
    except ValueError:
        # ValueError is raised if resolved_input_path is not within resolved_allowed_path
        return False


@app.get('/outputs/{file_name}')
def get_outputs(request: Request, file_name: str):
    assert os.pathsep not in file_name, "didn't think FastAPI would allow this; need to handle it properly then."
    file_path = OUTPUTS_PATH / file_name

    assert path_is_within(file_path, OUTPUTS_PATH), "FATAL ERROR: unexpected directory traversal would have occurred! Aborting."

    return FileResponse(file_path)


@app.get('/static/{file_name}')
def get_static(request: Request, file_name: str):
    assert os.pathsep not in file_name, "didn't think FastAPI would allow this; need to handle it properly then."
    file_path = STATIC_PATH / file_name

    assert path_is_within(file_path, STATIC_PATH), "FATAL ERROR: unexpected directory traversal would have occurred! Aborting."

    return FileResponse(file_path)


def main():
    logging.basicConfig()

    # Ensure output directory exists
    os.makedirs(OUTPUTS_PATH, exist_ok=True)

    print("🔥 Starting Orpheus-FASTAPI Server (CUDA)")
    print(f"    Serving OpenAI-compatible Speech API on {app.config.host}:{app.config.port}")
    print(f"    Using OpenAI compatible Orpheus text generation API at {app.config.api_url}")
    print(f"    Using output directory {OUTPUTS_PATH}")

    uvicorn.run(f"{__name__}:app", host="0.0.0.0", port=5005, reload=True, log_level='trace')

