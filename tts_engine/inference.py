import os
import sys
import requests
import json
import time
import wave
import numpy as np
import sounddevice as sd
import argparse
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Generator, Union, Tuple, AsyncGenerator
from dotenv import load_dotenv
import nltk
import multiprocessing
import logging

logger = logging.getLogger(__name__)

# Helper to ensure NLTK data is downloaded
_nltk_punkt_downloaded = False
def ensure_nltk_punkt():
    global _nltk_punkt_downloaded
    if not _nltk_punkt_downloaded:
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            print("NLTK 'punkt' tokenizer model not found. Downloading...")
            try:
                nltk.download('punkt', quiet=True)
                print("'punkt' model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading NLTK 'punkt' model: {e}")
                print("Sentence tokenization might be less accurate.")
        except LookupError: # Handle case where nltk_data path isn't configured
             print("NLTK 'punkt' model not found. Downloading...")
             try:
                 nltk.download('punkt', quiet=True)
                 print("'punkt' model downloaded successfully.")
             except Exception as e:
                 print(f"Error downloading NLTK 'punkt' model: {e}")
                 print("Sentence tokenization might be less accurate.")
        _nltk_punkt_downloaded = True

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

# Set a flag to avoid repeat messages
IS_RELOADER = is_reloader_process()
if not IS_RELOADER:
    os.environ['UVICORN_STARTED'] = 'true'

# Load environment variables from .env file
load_dotenv()

# Detect hardware capabilities and display information
import torch
import psutil

# Device selection with support for Apple Silicon MPS
# Define device globally for consistent use throughout
DEVICE = "cpu"  # Default to CPU, will be updated based on availability

# Detect if we're on a high-end system based on hardware capabilities
HIGH_END_GPU = False
APPLE_SILICON = False

# Check for Apple Silicon MPS support first
if torch.backends.mps.is_available():
    DEVICE = "mps"
    APPLE_SILICON = True
    
    # Get Apple Silicon details
    import platform
    import subprocess
    
    # Get chip model and memory
    chip_model = platform.processor()
    try:
        # Get memory info using sysctl
        mem_cmd = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
        if mem_cmd.returncode == 0:
            mem_bytes = int(mem_cmd.stdout.split(':')[1].strip())
            mem_gb = mem_bytes / (1024**3)
        else:
            mem_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        mem_gb = psutil.virtual_memory().total / (1024**3)
    
    # Detect high-end Apple Silicon (M1 Pro/Max/Ultra, M2 Pro/Max/Ultra, M3 Pro/Max/Ultra)
    if "Pro" in chip_model or "Max" in chip_model or "Ultra" in chip_model or mem_gb >= 32:
        HIGH_END_GPU = True
        if not IS_RELOADER:
            print(f"🍎 Hardware: High-end Apple Silicon detected")
            print(f"📊 Chip: {chip_model}")
            print(f"📊 RAM: {mem_gb:.2f} GB unified memory")
            print("🚀 Using high-performance Apple Silicon optimizations")
    else:
        if not IS_RELOADER:
            print(f"🍎 Hardware: Apple Silicon detected")
            print(f"📊 Chip: {chip_model}")
            print(f"📊 RAM: {mem_gb:.2f} GB unified memory")
            print("🚀 Using Apple Silicon optimizations")
            
# Then check for CUDA GPU
elif torch.cuda.is_available():
    DEVICE = "cuda"
    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    gpu_mem_gb = props.total_memory / (1024**3)
    compute_capability = f"{props.major}.{props.minor}"
    
    # Consider high-end if: large VRAM (≥16GB) OR high compute capability (≥8.0) OR large VRAM (≥12GB) with good CC (≥7.0)
    HIGH_END_GPU = (gpu_mem_gb >= 16.0 or 
                    props.major >= 8 or 
                    (gpu_mem_gb >= 12.0 and props.major >= 7))
        
    if HIGH_END_GPU:
        if not IS_RELOADER:
            print(f"🖥️ Hardware: High-end CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀 Using high-performance optimizations")
    else:
        if not IS_RELOADER:
            print(f"🖥️ Hardware: CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀 Using GPU-optimized settings")
else:
    # Get CPU info
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if not IS_RELOADER:
        print(f"🖥️ Hardware: CPU only (No GPU acceleration detected)")
        print(f"📊 CPU: {cpu_cores} cores, {cpu_threads} threads")
        print(f"📊 RAM: {ram_gb:.2f} GB")
        print("⚙️ Using CPU-optimized settings")

# Load configuration from environment variables without hardcoded defaults
# Critical settings - will log errors if missing
required_settings = ["ORPHEUS_API_URL"]
missing_settings = [s for s in required_settings if s not in os.environ]
if missing_settings:
    print(f"ERROR: Missing required environment variable(s): {', '.join(missing_settings)}")
    print("Please set them in .env file or environment. See .env.example for defaults.")

# API connection settings
API_URL = os.environ.get("ORPHEUS_API_URL")
if not API_URL:
    print("WARNING: ORPHEUS_API_URL not set. API calls will fail until configured.")

HEADERS = {
    "Content-Type": "application/json"
}

# Request timeout settings
try:
    REQUEST_TIMEOUT = int(os.environ.get("ORPHEUS_API_TIMEOUT", "120"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_API_TIMEOUT value, using 120 seconds as fallback")
    REQUEST_TIMEOUT = 120

# Model generation parameters from environment variables
try:
    MAX_TOKENS = int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_MAX_TOKENS value, using 8192 as fallback")
    MAX_TOKENS = 8192

try:
    TEMPERATURE = float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_TEMPERATURE value, using 0.6 as fallback")
    TEMPERATURE = 0.6

try:
    TOP_P = float(os.environ.get("ORPHEUS_TOP_P", "0.9"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_TOP_P value, using 0.9 as fallback")
    TOP_P = 0.9

# Repetition penalty is hardcoded to 1.1 which is the only stable value for quality output
REPETITION_PENALTY = 1.1

try:
    SAMPLE_RATE = int(os.environ.get("ORPHEUS_SAMPLE_RATE", "24000"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_SAMPLE_RATE value, using 24000 as fallback")
    SAMPLE_RATE = 24000

# Print loaded configuration only in the main process, not in the reloader
if not IS_RELOADER:
    print(f"Configuration loaded:")
    print(f"  API_URL: {API_URL}")
    print(f"  MAX_TOKENS: {MAX_TOKENS}")
    print(f"  TEMPERATURE: {TEMPERATURE}")
    print(f"  TOP_P: {TOP_P}")
    print(f"  REPETITION_PENALTY: {REPETITION_PENALTY}")

# Parallel processing settings
import multiprocessing

# Determine optimal settings based on hardware
CPU_CORES = multiprocessing.cpu_count()

# For Apple Silicon, use more aggressive settings depending on the model
if APPLE_SILICON:
    # Optimize for Apple Silicon based on memory
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb >= 64:  # High-memory M1 Max/Ultra, M2 Max/Ultra, M3 Max/Ultra (64GB+)
        NUM_WORKERS = max(8, min(CPU_CORES-2, 12))
        BATCH_SIZE = 64
        AUDIO_QUEUE_SIZE = 200
    elif ram_gb >= 32:  # Mid-range models (32GB)
        NUM_WORKERS = max(4, min(CPU_CORES-2, 8))
        BATCH_SIZE = 48
        AUDIO_QUEUE_SIZE = 150
    else:  # Base models
        NUM_WORKERS = max(2, min(CPU_CORES-1, 4))
        BATCH_SIZE = 32
        AUDIO_QUEUE_SIZE = 100
elif HIGH_END_GPU:  # High-end CUDA GPU
    NUM_WORKERS = 4
    BATCH_SIZE = 32
    AUDIO_QUEUE_SIZE = 100
else:  # Regular CUDA or CPU
    NUM_WORKERS = 2
    BATCH_SIZE = 16
    AUDIO_QUEUE_SIZE = 50

# Buffer size for audio processing
if APPLE_SILICON and psutil.virtual_memory().total >= (64 * 1024**3):  # 64GB+ RAM
    BUFFER_SIZE_MB = 4  # 4MB buffer
elif APPLE_SILICON or HIGH_END_GPU:
    BUFFER_SIZE_MB = 2  # 2MB buffer
else:
    BUFFER_SIZE_MB = 1  # 1MB buffer

BUFFER_MAX_SIZE = BUFFER_SIZE_MB * 1024 * 1024

# Maximum number of characters per batch for NLTK sentence splitting
try:
    MAX_BATCH_CHARS = int(os.environ.get("ORPHEUS_MAX_BATCH_CHARS", "600"))
    if MAX_BATCH_CHARS < 100 or MAX_BATCH_CHARS > 2000:
        print(f"WARNING: Invalid ORPHEUS_MAX_BATCH_CHARS value ({MAX_BATCH_CHARS}), should be between 100-2000. Using 600 as fallback.")
        MAX_BATCH_CHARS = 600
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_MAX_BATCH_CHARS value, using 600 as fallback")
    MAX_BATCH_CHARS = 600

# Crossfade duration in milliseconds for stitching audio batches
try:
    CROSSFADE_MS = int(os.environ.get("ORPHEUS_CROSSFADE_MS", "30"))
    if CROSSFADE_MS < 10 or CROSSFADE_MS > 200:
        print(f"WARNING: Invalid ORPHEUS_CROSSFADE_MS value ({CROSSFADE_MS}), should be between 10-200. Using 30 as fallback.")
        CROSSFADE_MS = 30
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_CROSSFADE_MS value, using 30 as fallback")
    CROSSFADE_MS = 30

# Helper function to generate equal power fade curves using sine/cosine
def generate_equal_power_fade_curves(num_samples):
    """Generate fade-out and fade-in curves using sine/cosine for equal power crossfading."""
    # Create a linear ramp from 0 to pi/2
    ramp = np.linspace(0, np.pi/2, num_samples)
    # Use sine for fade-out and cosine for fade-in to maintain equal power
    fade_out = np.sin(ramp)
    fade_in = np.cos(ramp)
    return fade_out, fade_in

# Define voices by language
ENGLISH_VOICES = ["tara", "kaya", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
FRENCH_VOICES = ["pierre", "amelie", "marie"]
GERMAN_VOICES = ["jana", "thomas", "max"]
KOREAN_VOICES = ["유나", "준서"]
HINDI_VOICES = ["ऋतिका"]
MANDARIN_VOICES = ["长乐", "白芷"]
SPANISH_VOICES = ["javi", "sergio", "maria"]
ITALIAN_VOICES = ["pietro", "giulia", "carlo"]

# Combined list for API compatibility
AVAILABLE_VOICES = (
    ENGLISH_VOICES + 
    FRENCH_VOICES + 
    GERMAN_VOICES + 
    KOREAN_VOICES + 
    HINDI_VOICES + 
    MANDARIN_VOICES + 
    SPANISH_VOICES + 
    ITALIAN_VOICES
)
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Map voices to languages for the UI
VOICE_TO_LANGUAGE = {}
VOICE_TO_LANGUAGE.update({voice: "english" for voice in ENGLISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "french" for voice in FRENCH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "german" for voice in GERMAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "korean" for voice in KOREAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "hindi" for voice in HINDI_VOICES})
VOICE_TO_LANGUAGE.update({voice: "mandarin" for voice in MANDARIN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "spanish" for voice in SPANISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "italian" for voice in ITALIAN_VOICES})

# Languages list for the UI
AVAILABLE_LANGUAGES = ["english", "french", "german", "korean", "hindi", "mandarin", "spanish", "italian"]

# Import the unified token handling from speechpipe
from .speechpipe import turn_token_into_id, CUSTOM_TOKEN_PREFIX

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]

# Performance monitoring
class PerformanceMonitor:
    """Track and report performance metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = 2.0  # seconds
        
    def add_tokens(self, count: int = 1) -> None:
        self.token_count += count
        self._check_report()
        
    def add_audio_chunk(self) -> None:
        self.audio_chunks += 1
        self._check_report()
        
    def _check_report(self) -> None:
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
            
    def report(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed < 0.001:
            return
            
        tokens_per_sec = self.token_count / elapsed
        chunks_per_sec = self.audio_chunks / elapsed
        
        # Estimate audio duration based on audio chunks (each chunk is ~0.085s of audio)
        est_duration = self.audio_chunks * 0.085
        
        # print(f"Progress: {tokens_per_sec:.1f} tokens/sec, est. {est_duration:.1f}s audio generated, {self.token_count} tokens, {self.audio_chunks} chunks in {elapsed:.1f}s")

# Create global performance monitor
perf_monitor = PerformanceMonitor()

# --- Read System Prompt --- 
SYSTEM_PROMPT_CONTENT = None
PAST_CONTEXT = None # For long texts, we are going to store the past context in this variable and pass it to the model as a system prompt.
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "System_Prompt.md")

try:
    if os.path.exists(SYSTEM_PROMPT_PATH):
        with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            SYSTEM_PROMPT_CONTENT = f.read()
            if not IS_RELOADER:
                logger.info(f"Successfully loaded system prompt from {SYSTEM_PROMPT_PATH}")
    else:
        if not IS_RELOADER:
            logger.warning(f"System prompt file not found at {SYSTEM_PROMPT_PATH}. Proceeding without system prompt.")
except Exception as e:
    if not IS_RELOADER:
        logger.error(f"Error reading system prompt file {SYSTEM_PROMPT_PATH}: {e}")
# --- End System Prompt Read --- 

def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens.
       Restored based on backup file.
    """
    # Validate voice and provide fallback
    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    # Format similar to original backup
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers required by Orpheus
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    
    return f"{special_start}{formatted_prompt}{special_end}" # Restored original format

def generate_tokens_from_api(prompt: str, voice: str = DEFAULT_VOICE, temperature: float = TEMPERATURE, 
                           top_p: float = TOP_P, max_tokens: int = MAX_TOKENS, 
                           repetition_penalty: float = REPETITION_PENALTY,
                           context_prompt: Optional[str] = None) -> Generator[str, None, None]:
    """Generate tokens from text using OpenAI-compatible API (prompt format).
       Uses the separate 'system_prompt' payload field for system/context info if available.
    """
    start_time = time.time()
    # Format the main part of the user prompt (e.g., <|audio|>voice: text<|eot_id|>)
    formatted_user_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for user prompt (voice: {voice}): {prompt[:80]}...")

    # Combine system prompt and context prompt for the dedicated field
    system_and_context = ""
    if SYSTEM_PROMPT_CONTENT:
        system_and_context += SYSTEM_PROMPT_CONTENT
        print("Using system prompt.")
    if context_prompt:
        # Add clear separation if both system and context exist
        if system_and_context:
            system_and_context += "\n\n[Previous context:]\n"
        else: # Only context exists
            system_and_context += "[Previous context:]\n"
        system_and_context += context_prompt
        print("Using context prompt.")

    # Create the request payload 
    payload = {
        "prompt": formatted_user_prompt, # Main user prompt here
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }
    
    # Add the dedicated system_prompt field if we have content for it
    if system_and_context:
        payload["system"] = system_and_context
        print(f"Sending system_prompt field: {system_and_context[:100]}...")

    # Add model field - optional but good practice
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
    if model_name:
        payload["model"] = model_name
    
    # Session for connection pooling and retry logic
    session = requests.Session()
    
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # Make the API request
            response = session.post(
                API_URL, 
                headers=HEADERS, 
                json=payload, 
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"API Error ({response.status_code}): {error_detail}")
                # Removed the messages vs prompt check as we now use prompt
                if response.status_code >= 500:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                return
            
            # Process the streamed response (parsing logic remains the same, checking for 'text')
            token_counter = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                token_chunk = choice.get('text', '') # Expecting 'text' for /v1/completions
                                
                                if token_chunk and token_chunk.startswith("<custom_token_") and token_chunk.endswith(">"):
                                    token_counter += 1
                                    perf_monitor.add_tokens()
                                    yield token_chunk

                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON stream data: {e}")
                            continue
            
            generation_time = time.time() - start_time
            tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
            print(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            return
            
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached for timeout. Token generation failed.")
                return
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error connecting to API at {API_URL}: {e}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached for connection error. Token generation failed.")
                return

# The turn_token_into_id function is now imported from speechpipe.py
# This eliminates duplicate code and ensures consistent behavior

def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio with performance monitoring."""
    # Import here to avoid circular imports
    from .speechpipe import convert_to_audio as orpheus_convert_to_audio
    start_time = time.time()
    result = orpheus_convert_to_audio(multiframe, count)
    
    if result is not None:
        perf_monitor.add_audio_chunk()
        
    return result

async def tokens_decoder(token_gen) -> AsyncGenerator[bytes, None]:
    """Simplified token decoder with early first-chunk processing for lower latency."""
    buffer = []
    count = 0
    
    # Use different thresholds for first chunk vs. subsequent chunks
    first_chunk_processed = False
    min_frames_first = 7  # Process after just 7 tokens for first chunk (ultra-low latency)
    min_frames_subsequent = 28  # Default for reliability after first chunk (4 chunks of 7)
    process_every = 7  # Process every 7 tokens (standard for Orpheus model)
    
    start_time = time.time()
    last_log_time = start_time
    token_count = 0
    
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            # Add to buffer using simple append (reliable method)
            buffer.append(token)
            count += 1
            token_count += 1
            
            # Log throughput periodically
            # current_time = time.time()
            # if current_time - last_log_time > 5.0:  # Every 5 seconds
            #     elapsed = current_time - start_time
            #     if elapsed > 0:
            #         print(f"Token processing rate: {token_count/elapsed:.1f} tokens/second")
            #     last_log_time = current_time
            
            # Different processing paths based on whether first chunk has been processed
            if not first_chunk_processed:
                # For first audio output, process as soon as we have enough tokens for one chunk
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    
                    # Process the first chunk for immediate audio feedback
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True  # Mark first chunk as processed
                        yield audio_samples
            else:
                # For subsequent chunks, use standard processing with larger batch
                if count % process_every == 0 and count >= min_frames_subsequent:
                    # Use simple slice operation - reliable and correct
                    buffer_to_proc = buffer[-min_frames_subsequent:]
                    
                    # Debug output to help diagnose issues
                    # if count % 28 == 0:
                        # print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")
                    
                    # Process the tokens
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """Optimized synchronous wrapper with parallel processing and efficient file I/O."""
    # Use hardware-optimized queue size
    queue_size = AUDIO_QUEUE_SIZE
    audio_queue = queue.Queue(maxsize=queue_size)
    audio_segments = []
    
    # If output_file is provided, prepare WAV file with buffered I/O
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    # Use optimized batch processing
    batch_size = BATCH_SIZE
    
    # Thread synchronization for proper completion detection
    producer_done_event = threading.Event()
    producer_started_event = threading.Event()
    
    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        batch = []
        for token in syn_token_gen:
            batch.append(token)
            if len(batch) >= batch_size:
                for t in batch:
                    yield t
                batch = []
        # Process any remaining tokens in the final batch
        for t in batch:
            yield t

    async def async_producer():
        # Track performance with more granular metrics
        start_time = time.time()
        chunk_count = 0
        last_log_time = start_time
        
        try:
            # Signal that producer has started processing
            producer_started_event.set()
            
            async for audio_chunk in tokens_decoder(async_token_gen()):
                # Process each audio chunk from the decoder
                if audio_chunk:
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    # Log performance periodically
                    # current_time = time.time()
                    # if current_time - last_log_time >= 3.0:  # Every 3 seconds
                    #     elapsed = current_time - last_log_time
                    #     if elapsed > 0:
                    #         recent_chunks = chunk_count
                    #         chunks_per_sec = recent_chunks / elapsed
                    #         print(f"Audio generation rate: {chunks_per_sec:.2f} chunks/second")
                    #     last_log_time = current_time 
                    #     # Reset chunk counter for next interval
                    #     chunk_count = 0
        except Exception as e:
            print(f"Error in token processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Always signal completion, even if there was an error
            print("Producer completed - setting done event")
            producer_done_event.set()
            # Add sentinel to queue to signal end of stream
            audio_queue.put(None)

    def run_async():
        """Run the async producer in its own thread"""
        asyncio.run(async_producer())

    # Use a separate thread with higher priority for producer
    thread = threading.Thread(target=run_async, name="TokenProcessor")
    thread.daemon = True  # Allow thread to be terminated when main thread exits
    thread.start()
    
    # Wait for producer to actually start before proceeding
    # This avoids race conditions where we might try to read from an empty queue
    # before the producer has had a chance to add anything
    producer_started_event.wait(timeout=5.0)
    
    # Use hardware-optimized buffer size
    write_buffer = bytearray()
    buffer_max_size = BUFFER_MAX_SIZE
    
    # Keep track of the last time we checked for completion
    last_check_time = time.time()
    check_interval = 1.0  # Check producer status every second
    
    # Process audio chunks until we're done
    while True:
        try:
            # Get the next audio chunk with a short timeout
            # This allows us to periodically check status and handle other events
            audio = audio_queue.get(timeout=0.1)
            
            # None marker indicates end of stream
            if audio is None:
                print("Received end-of-stream marker")
                break
            
            # Store the audio segment for return value
            audio_segments.append(audio)
            
            # Write to file if needed
            if wav_file:
                write_buffer.extend(audio)
                
                # Flush buffer if it's large enough
                if len(write_buffer) >= buffer_max_size:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()  # Reset buffer
        
        except queue.Empty:
            # No data available right now
            current_time = time.time()
            
            # Periodically check if producer is done
            if current_time - last_check_time > check_interval:
                last_check_time = current_time
                
                # If producer is done and queue is empty, we're finished
                if producer_done_event.is_set() and audio_queue.empty():
                    print("Producer done and queue empty - finishing consumer")
                    break
                
                # Flush buffer periodically even if not full
                if wav_file and len(write_buffer) > 0:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()  # Reset buffer
    
    # Extra safety check - ensure thread is done
    if thread.is_alive():
        print("Waiting for token processor thread to complete...")
        thread.join(timeout=10.0)
        if thread.is_alive():
            print("WARNING: Token processor thread did not complete within timeout")
    
    # Final flush of any remaining data
    if wav_file and len(write_buffer) > 0:
        print(f"Final buffer flush: {len(write_buffer)} bytes")
        wav_file.writeframes(write_buffer)
    
    # Close WAV file if opened
    if wav_file:
        wav_file.close()
        if output_file:
            print(f"Audio saved to {output_file}")
    
    # Calculate and print detailed performance metrics
    if audio_segments:
        total_bytes = sum(len(segment) for segment in audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)  # 2 bytes per sample at 24kHz
        total_time = time.time() - perf_monitor.start_time
        realtime_factor = duration / total_time if total_time > 0 else 0
        
        print(f"Generated {len(audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        
        if realtime_factor < 1.0:
            print("⚠️ Warning: Generation is slower than realtime")
        else:
            print(f"✓ Generation is {realtime_factor:.1f}x faster than realtime")
    
    return audio_segments

def stream_audio(audio_buffer):
    """Stream audio buffer to output device with error handling."""
    if audio_buffer is None or len(audio_buffer) == 0:
        return
    
    try:
        # Convert bytes to NumPy array (16-bit PCM)
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
        
        # Normalize to float in range [-1, 1] for playback
        audio_float = audio_data.astype(np.float32) / 32767.0
        
        # Play the audio with proper device selection and error handling
        sd.play(audio_float, SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")

import re
import numpy as np
from io import BytesIO
import wave

# Map for NLTK language names (add more as needed/supported by punkt)
NLTK_LANG_MAP = {
    "english": "english",
    "french": "french",
    "german": "german",
    "spanish": "spanish",
    "italian": "italian",
    # Add other languages if punkt models exist for them
}

def split_text_into_sentences(text: str, language: str = "english", max_chars_per_segment: int = MAX_BATCH_CHARS) -> List[str]:
    """Split text into sentences using NLTK for better accuracy."""
    ensure_nltk_punkt() # Make sure model is available

    nltk_lang = NLTK_LANG_MAP.get(language, "english") # Default to English if mapped lang not found
    print(f"Splitting text into sentences using NLTK for language: {nltk_lang}")
    
    try:
        # Attempt to use the specified language model
        sentences = nltk.sent_tokenize(text, language=nltk_lang)
        print(f"Successfully tokenized using NLTK for language: {nltk_lang}")
    except Exception as e:
        # Fallback to default English model if specific language model fails or isn't available
        print(f"Warning: Could not use NLTK tokenizer for language '{nltk_lang}'. Falling back to English. Error: {e}")
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as inner_e:
            # If NLTK fails completely, fallback to a very basic split as a last resort
            print(f"ERROR: NLTK sentence tokenization failed entirely: {inner_e}. Using basic fallback.")
            sentences = text.split('. ') # Very basic fallback

    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Create segments that respect the max_chars_per_segment limit
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        # If a single sentence exceeds the limit, we need to split it
        if len(sentence) > max_chars_per_segment:
            # If we have a current segment, add it first
            if current_segment:
                segments.append(current_segment)
                current_segment = ""
            
            # Split the long sentence into chunks
            words = sentence.split()
            current_chunk = ""
            
            for word in words:
                # If adding this word would exceed the limit, start a new chunk
                if len(current_chunk) + len(word) + 1 > max_chars_per_segment:
                    if current_chunk:
                        segments.append(current_chunk)
                    current_chunk = word
                else:
                    # Add word to current chunk with a space if needed
                    current_chunk = f"{current_chunk} {word}" if current_chunk else word
            
            # Add the last chunk if it exists
            if current_chunk:
                segments.append(current_chunk)
        else:
            # Check if adding this sentence would exceed the limit
            if current_segment:
                potential_length = len(current_segment) + 1 + len(sentence)
                if potential_length > max_chars_per_segment:
                    segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment = f"{current_segment} {sentence}"
            else:
                current_segment = sentence
    
    # Add the last segment if it exists
    if current_segment:
        segments.append(current_segment)
    
    # Verify that no segment exceeds the limit
    for i, segment in enumerate(segments):
        if len(segment) > max_chars_per_segment:
            print(f"Warning: Segment {i} exceeds max_chars_per_segment ({len(segment)} > {max_chars_per_segment})")
            # Split the segment into smaller chunks
            words = segment.split()
            new_segments = []
            current_chunk = ""
            
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chars_per_segment:
                    if current_chunk:
                        new_segments.append(current_chunk)
                    current_chunk = word
                else:
                    current_chunk = f"{current_chunk} {word}" if current_chunk else word
            
            if current_chunk:
                new_segments.append(current_chunk)
            
            # Replace the long segment with the new segments
            segments[i:i+1] = new_segments
    
    print(f"Split text into {len(segments)} segments with max length {max_chars_per_segment}")
    return segments

def cleanup_between_batches():
    """Reset all state between batch processing."""
    # Reset the performance monitor
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    # Import here to avoid circular imports
    from .speechpipe import reset_state
    # Reset the speechpipe state
    reset_state()
    
    # Force garbage collection
    import gc
    gc.collect()

def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None, temperature=TEMPERATURE, 
                     top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=None, 
                     use_batching=True, max_batch_chars=MAX_BATCH_CHARS):
    """Generate speech from text using Orpheus model with performance optimizations and NLTK splitting."""
    print(f"Starting speech generation for '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"Using voice: {voice}, GPU acceleration: {'Yes (High-end)' if HIGH_END_GPU else 'Yes' if torch.cuda.is_available() or torch.backends.mps.is_available() else 'No'}")
    
    # Reset performance monitor at start
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    all_audio_segments = [] # To store the final small chunks for return/streaming

    # Determine language for splitting
    generation_language = VOICE_TO_LANGUAGE.get(voice, "english")

    # For shorter text or disabled batching, use the standard non-batched approach
    if not use_batching or len(prompt) < max_batch_chars:
        print("Processing text as a single batch.")
        # Note: repetition_penalty is ignored (uses hardcoded 1.1)
        all_audio_segments = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt, 
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY # Fixed value
            ),
            output_file=output_file # Pass file handle if saving
        )
        
        if output_file:
             pass # File writing is handled internally by tokens_decoder_sync
        else:
             pass # Segments are already in all_audio_segments

    # For longer text, use sentence-based batching with NLTK and crossfading
    else:
        print(f"Using sentence-based batching for text with {len(prompt)} characters (limit: {max_batch_chars})")
        
        # Split the text into segments using NLTK, passing the limit
        segments = split_text_into_sentences(prompt, language=generation_language, max_chars_per_segment=max_batch_chars)
        print(f"Split text into {len(segments)} segments using NLTK ({generation_language}).")
        
        # Process each segment and collect audio segments
        all_complete_batch_audio = [] # Store complete audio (bytes) for each batch
        batch_temp_files = []
        all_results = [(None, None, None, None)] * len(segments) # Pre-allocate results list
        
        # Determine if we should use parallel processing
        use_parallel = APPLE_SILICON and len(segments) > 1 and psutil.virtual_memory().total >= (32 * 1024**3)
        
        if use_parallel:
            print(f"Using parallel processing with {NUM_WORKERS} workers for {len(segments)} segments")
            
            # Define a function to process a single segment
            def process_segment(segment_text, index, previous_segment_text):
                print(f"Starting parallel processing of segment {index+1}/{len(segments)} ({len(segment_text)} characters)")
                
                # Clean up state before processing
                cleanup_between_batches()
                
                # --- Prepare context from previous segment --- 
                context_for_next = ""
                if previous_segment_text:
                    try:
                        # Extract last sentence as context
                        # Make sure NLTK data is available (should be loaded globally)
                        ensure_nltk_punkt() 
                        prev_sentences = nltk.sent_tokenize(previous_segment_text)
                        if prev_sentences:
                             # Use only the last sentence as context
                             context_for_next = prev_sentences[-1].strip()
                             print(f"Segment {index+1}: Using last sentence of previous segment as context.")
                    except Exception as e:
                         logger.warning(f"Segment {index+1}: Could not extract context from previous segment: {e}")
                # --- End context preparation --- 
                
                # Create a temporary file if needed
                temp_file = None
                if output_file:
                    temp_file = f"outputs/temp_batch_{index}_{int(time.time())}.wav"
                    # Note: Appending to batch_temp_files needs thread-safety or post-processing
                
                # Process the segment, passing the context
                segments_output = tokens_decoder_sync(
                    generate_tokens_from_api(
                        prompt=segment_text,
                        voice=voice,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        repetition_penalty=REPETITION_PENALTY,
                        context_prompt=context_for_next # Pass context here
                    ),
                    output_file=temp_file
                )
                
                # Combine the audio fragments
                complete_audio = b"".join(segments_output)
                print(f"Completed parallel processing of segment {index+1}")
                
                return index, complete_audio, segments_output, temp_file
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Submit all segments for processing, passing previous segment text
                future_to_segment = {
                    executor.submit(process_segment, 
                                    segment, 
                                    i, 
                                    segments[i-1] if i > 0 else None): # Pass previous segment
                    i for i, segment in enumerate(segments)
                }
                
                # Process segments as they complete
                temp_batch_files_collector = {} # Collect temp files safely
                for future in as_completed(future_to_segment):
                    try:
                        idx, complete_audio, segments_output, temp_file = future.result()
                        # Store the results in the pre-allocated list using the index
                        all_results[idx] = (idx, complete_audio, segments_output, temp_file)
                        if temp_file:
                           temp_batch_files_collector[idx] = temp_file
                    except Exception as e:
                        idx = future_to_segment[future] # Get index from map
                        print(f"Error processing segment {idx+1}: {e}")
                        # Store error indication if needed, e.g., all_results[idx] = (idx, None, None, None)
                
                # Extract results in original order from the pre-allocated list
                all_complete_batch_audio = [audio for idx, audio, _, _ in all_results if audio is not None]
                batch_temp_files = [temp_batch_files_collector[i] for i in sorted(temp_batch_files_collector) if temp_batch_files_collector[i] is not None]
                
                if not output_file:
                    # Flatten the segments in original order
                    temp_segments = []
                    for idx, _, segments_output, _ in all_results:
                        if segments_output is not None:
                           temp_segments.extend(segments_output)
                    all_audio_segments = temp_segments
        else:
            # Process segments sequentially (original logic, but add context passing)
            for i, segment in enumerate(segments):
                print(f"Processing segment {i+1}/{len(segments)} ({len(segment)} characters)")
                
                # Clean up state between batches
                cleanup_between_batches()
                
                # --- Prepare context from previous segment --- 
                context_for_next = ""
                if i > 0 and segments[i-1]: # Check if previous segment exists
                    try:
                        ensure_nltk_punkt()
                        prev_sentences = nltk.sent_tokenize(segments[i-1])
                        if prev_sentences:
                             context_for_next = prev_sentences[-1].strip()
                             print(f"Segment {i+1}: Using last sentence of previous segment as context.")
                    except Exception as e:
                         logger.warning(f"Segment {i+1}: Could not extract context from previous segment: {e}")
                # --- End context preparation --- 

                # Create a temporary file ONLY if a final output file is requested
                temp_output_file_for_batch = None
                if output_file:
                    temp_output_file_for_batch = f"outputs/temp_batch_{i}_{int(time.time())}.wav"
                    batch_temp_files.append(temp_output_file_for_batch)
                
                # Generate speech for this segment, passing context
                batch_segments = tokens_decoder_sync(
                    generate_tokens_from_api(
                        prompt=segment,
                        voice=voice,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        repetition_penalty=REPETITION_PENALTY,
                        context_prompt=context_for_next # Pass context here
                    ),
                    output_file=temp_output_file_for_batch
                )
                
                # Combine the small segments from this batch into one bytes object
                complete_batch_audio_bytes = b"".join(batch_segments)
                all_complete_batch_audio.append(complete_batch_audio_bytes)
                
                # Also keep track of small segments if not writing to file and no crossfading needed later
                if not output_file:
                     all_audio_segments.extend(batch_segments)
            
            # Explicitly clear segments list after processing all segments
            segments = None
            import gc
            gc.collect()

        # --- Post-Batch Processing --- 
        if output_file:
            # If an output file was requested, stitch together the temporary batch files
            if batch_temp_files:
                stitch_wav_files(batch_temp_files, output_file)
                # Clean up temporary files
                for temp_file in batch_temp_files:
                    try: os.remove(temp_file) 
                    except Exception as e: print(f"Warning: Could not remove temp file {temp_file}: {e}")
            # In file output mode, the final return value isn't the audio data itself
            all_audio_segments = [] # Clear segments as data is in the file
        
        elif len(all_complete_batch_audio) > 1:
             # If NO output file AND more than one batch was processed, apply crossfading
             print(f"Applying in-memory crossfade to {len(all_complete_batch_audio)} audio batches.")
             try:
                 # Convert bytes to numpy arrays
                 batch_arrays = [np.frombuffer(b, dtype=np.int16) for b in all_complete_batch_audio if b] # Filter empty
                 
                 if len(batch_arrays) > 1:
                     # Apply crossfading logic with optimizations for high-memory systems
                     if APPLE_SILICON and psutil.virtual_memory().total >= (64 * 1024**3):
                         # Pre-allocate the final array to avoid repeated concatenations
                         # First calculate total length of all arrays minus crossfade regions
                         crossfade_samples = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
                         total_samples = sum(len(arr) for arr in batch_arrays)
                         # Subtract overlapping regions
                         total_samples -= crossfade_samples * (len(batch_arrays) - 1)
                         
                         # Pre-allocate the final array
                         print(f"Pre-allocating array for {total_samples} samples")
                         final_audio_np = np.zeros(total_samples, dtype=np.int16)
                         
                         # Fill the array with crossfaded audio
                         write_position = 0
                         
                         for i, audio_np in enumerate(batch_arrays):
                             if i == 0:
                                 # First segment - copy directly
                                 final_audio_np[:len(audio_np)] = audio_np
                                 write_position += len(audio_np) - crossfade_samples
                             else:
                                 # For other segments, apply crossfade
                                 # Generate equal power fade curves
                                 fade_out, fade_in = generate_equal_power_fade_curves(crossfade_samples)
                                 
                                 # Create crossfade region
                                 prev_end = write_position
                                 crossfade_region = (final_audio_np[prev_end:prev_end+crossfade_samples] * fade_out + 
                                                    audio_np[:crossfade_samples] * fade_in).astype(np.int16)
                                 
                                 # Write crossfade region
                                 final_audio_np[prev_end:prev_end+crossfade_samples] = crossfade_region
                                 
                                 # Write remainder of current segment
                                 next_end = prev_end + crossfade_samples + len(audio_np) - crossfade_samples
                                 final_audio_np[prev_end+crossfade_samples:next_end] = audio_np[crossfade_samples:]
                                 
                                 # Update write position
                                 write_position = next_end
                     else:
                         # Use standard approach for systems with less memory
                         final_audio_np = np.array([], dtype=np.int16)
                         # Use the constant for crossfade duration
                         crossfade_samples = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
                         print(f"Applying {CROSSFADE_MS}ms crossfade ({crossfade_samples} samples)")

                         for i, audio_np in enumerate(batch_arrays):
                             if i == 0:
                                 final_audio_np = audio_np
                             else:
                                 # Apply crossfade
                                 prev_audio_np = final_audio_np
                                 current_audio_np = audio_np
                                 
                                 if len(prev_audio_np) >= crossfade_samples and len(current_audio_np) >= crossfade_samples:
                                     # Generate equal power fade curves
                                     fade_out, fade_in = generate_equal_power_fade_curves(crossfade_samples)
                                     
                                     crossfade_region = (prev_audio_np[-crossfade_samples:] * fade_out + 
                                                        current_audio_np[:crossfade_samples] * fade_in).astype(np.int16)
                                     
                                     final_audio_np = np.concatenate([
                                         prev_audio_np[:-crossfade_samples], 
                                         crossfade_region, 
                                         current_audio_np[crossfade_samples:]
                                     ])
                                 else:
                                     # Segments too short for crossfade, concatenate directly
                                     print(f"Warning: Segments too short for crossfade between batch {i-1} and {i}. Concatenating.")
                                     final_audio_np = np.concatenate([prev_audio_np, current_audio_np])
                                     
                     # Convert final numpy array back to bytes and wrap in a list
                     all_audio_segments = [final_audio_np.tobytes()]
                     print("In-memory crossfading complete.")
                     
                     # Clear batch arrays to free memory
                     batch_arrays = None
                     all_complete_batch_audio = None
                     gc.collect()
                 else:
                      # Only one valid batch array, use the original segments
                      print("Only one batch generated after filtering, no crossfading needed.")
                      pass 
             except Exception as e:
                  print(f"ERROR during in-memory crossfading: {e}. Returning raw concatenated segments.")
                  # Fallback: return the original potentially discontinuous segments if crossfade fails
                  all_audio_segments = [b"".join(all_audio_segments)] # Combine all small chunks

    # --- Final Reporting --- 
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate combined duration from the final segments to be returned
    if all_audio_segments:
        # Handle if crossfading resulted in a single large chunk
        if len(all_audio_segments) == 1:
             total_bytes = len(all_audio_segments[0])
        else: # Original chunked segments
             total_bytes = sum(len(segment) for segment in all_audio_segments)
        
        if total_bytes > 0:
             duration = total_bytes / (2 * SAMPLE_RATE) # 2 bytes per sample
             print(f"Generated {len(all_audio_segments)} final audio segment(s)") # Correctly reports 1 segment after crossfade
             print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
             if total_time > 0:
                  realtime_factor = duration / total_time
                  print(f"Realtime factor: {realtime_factor:.2f}x")
                  if realtime_factor < 1.0:
                      print("⚠️ Warning: Generation is slower than realtime")
                  else:
                      print(f"✓ Generation is {realtime_factor:.1f}x faster than realtime")
             else:
                  print("Generation time was negligible.")
        else:
            print("Warning: No audio data generated.")
            
    print(f"Total speech generation completed in {total_time:.2f} seconds")
    
    # Return the final audio segments (either original chunks or one combined chunk after crossfade)
    return all_audio_segments

def stitch_wav_files(input_files, output_file):
    """Stitch multiple WAV files together with crossfading for smooth transitions."""
    if not input_files:
        return
        
    print(f"Stitching {len(input_files)} WAV files together with {CROSSFADE_MS}ms crossfade")
    
    # If only one file, just copy it
    if len(input_files) == 1:
        import shutil
        shutil.copy(input_files[0], output_file)
        print(f"Only one input file, copied directly to {output_file}")
        return
    
    # Convert crossfade_ms to samples using the constant
    crossfade_samples = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
    print(f"Using {crossfade_samples} samples for crossfade at {SAMPLE_RATE}Hz")
    
    # Build the final audio in memory with crossfades
    final_audio = np.array([], dtype=np.int16)
    first_params = None
    
    # Standard WAV parameters to enforce
    standard_params = {
        'nchannels': 1,
        'sampwidth': 2,
        'framerate': SAMPLE_RATE
    }
    
    for i, input_file in enumerate(input_files):
        try:
            with wave.open(input_file, 'rb') as wav:
                # Get current file parameters
                current_params = wav.getparams()
                
                # Check and standardize parameters
                if first_params is None:
                    first_params = current_params
                    # Verify first file meets our standards
                    if (current_params.nchannels != standard_params['nchannels'] or
                        current_params.sampwidth != standard_params['sampwidth'] or
                        current_params.framerate != standard_params['framerate']):
                        print(f"Warning: First WAV file {input_file} has non-standard parameters. Converting to standard format.")
                elif current_params != first_params:
                    print(f"Warning: WAV file {input_file} has different parameters. Converting to standard format.")
                
                # Read frames and convert to numpy array
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                
                if i == 0:
                    # First segment - use as is
                    final_audio = audio
                else:
                    # Apply crossfade with previous segment
                    if len(final_audio) >= crossfade_samples and len(audio) >= crossfade_samples:
                        # Generate equal power fade curves
                        fade_out, fade_in = generate_equal_power_fade_curves(crossfade_samples)
                        
                        # Apply crossfade
                        crossfade_region = (final_audio[-crossfade_samples:] * fade_out + 
                                           audio[:crossfade_samples] * fade_in).astype(np.int16)
                        
                        # Combine: original without last crossfade_samples + crossfade + new without first crossfade_samples
                        final_audio = np.concatenate([final_audio[:-crossfade_samples], 
                                                    crossfade_region, 
                                                    audio[crossfade_samples:]])
                    else:
                        # One segment too short for crossfade, just append
                        print(f"Segment {i} too short for crossfade, concatenating directly")
                        final_audio = np.concatenate([final_audio, audio])
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            if i == 0:
                raise  # Critical failure if first file fails
    
    # Write the final audio data to the output file
    try:
        with wave.open(output_file, 'wb') as output_wav:
            if first_params is None:
                raise ValueError("No valid WAV files were processed")
                
            # Use standard parameters for output
            output_wav.setnchannels(standard_params['nchannels'])
            output_wav.setsampwidth(standard_params['sampwidth'])
            output_wav.setframerate(standard_params['framerate'])
            output_wav.writeframes(final_audio.tobytes())
        
        print(f"Successfully stitched audio to {output_file} with crossfading")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")
        raise

def list_available_voices():
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")
    
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech using Orpheus-FASTAPI")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, 
                       help="Repetition penalty (fixed at 1.1 for stable generation - parameter kept for compatibility)")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Use text from command line or prompt user
    prompt = args.text
    if not prompt:
        if len(sys.argv) > 1 and sys.argv[1] not in ("--voice", "--output", "--temperature", "--top_p", "--repetition_penalty"):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."
    
    # Default output file if none provided
    output_file = args.output
    if not output_file:
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        # Generate a filename based on the voice and a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")
    
    # Generate speech
    start_time = time.time()
    audio_segments = generate_speech_from_api(
        prompt=prompt,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_file=output_file
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    main()
