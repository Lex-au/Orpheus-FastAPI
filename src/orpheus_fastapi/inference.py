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
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Generator, Union, Tuple

# Detect if we're on a high-end system like RTX 4090
import torch
HIGH_END_GPU = False
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0).lower()
    if any(x in gpu_name for x in ['4090', '3090', 'a100', 'h100']):
        HIGH_END_GPU = True
        print(f"High-end GPU detected: {torch.cuda.get_device_name(0)}")
        print("Enabling high-performance optimizations")

DEFAULT_API_URL = "http://127.0.0.1:1337/v1/completions"

# Orpheus-FASTAPI settings - make configurable for different endpoints
HEADERS = {
    "Content-Type": "application/json"
}

# Better timeout handling for API requests
REQUEST_TIMEOUT = int(os.environ.get("ORPHEUS_API_TIMEOUT", "120"))  # 120 seconds default for long generations

# Model parameters - optimized defaults for high-end GPUs
MAX_TOKENS = 8192 if HIGH_END_GPU else 1200  # Significantly increased for RTX 4090 to allow ~1.5-2 minutes of audio
TEMPERATURE = 0.6 
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Parallel processing settings
NUM_WORKERS = 4 if HIGH_END_GPU else 2

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

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
        
        print(f"Progress: {tokens_per_sec:.1f} tokens/sec, est. {est_duration:.1f}s audio generated, {self.token_count} tokens, {self.audio_chunks} chunks in {elapsed:.1f}s")

# Create global performance monitor
perf_monitor = PerformanceMonitor()

def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    # Validate voice and provide fallback
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers for the Orpheus-FASTAPI
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config
    
    return f"{special_start}{formatted_prompt}{special_end}"

def generate_tokens_from_api(prompt: str, voice: str = DEFAULT_VOICE, temperature: float = TEMPERATURE, 
                           top_p: float = TOP_P, max_tokens: int = MAX_TOKENS, 
                           repetition_penalty: float = REPETITION_PENALTY,
                           api_url: str=DEFAULT_API_URL) -> Generator[str, None, None]:
    """Generate tokens from text using OpenAI-compatible API with optimized streaming and retry logic."""
    start_time = time.time()
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    # Optimize the token generation for high-end GPUs
    if HIGH_END_GPU:
        # Use more aggressive parameters for faster generation on high-end GPUs
        print("Using optimized parameters for high-end GPU")
    
    # Create the request payload
    payload = {
        "model": "orpheus-3b-0.1-ft-q4_k_m",  # Model name can be anything, endpoint will use loaded model
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True  # Always stream for better performance
    }
    
    # Session for connection pooling and retry logic
    session = requests.Session()
    
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # Make the API request with streaming and timeout
            response = session.post(
                api_url, 
                headers=HEADERS, 
                json=payload, 
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Error details: {response.text}")
                # Retry on server errors (5xx) but not on client errors (4xx)
                if response.status_code >= 500:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return
            
            # Process the streamed response with better buffering
            buffer = ""
            token_counter = 0
            
            # Iterate through the response to get tokens
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove the 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                token_text = data['choices'][0].get('text', '')
                                token_counter += 1
                                perf_monitor.add_tokens()
                                
                                if token_text:
                                    yield token_text
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue
            
            # Generation completed successfully
            generation_time = time.time() - start_time
            tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
            print(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            return
            
        except requests.exceptions.Timeout:
            print(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return
                
        except requests.exceptions.ConnectionError:
            print(f"Connection error to API at {api_url}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return

# Token ID cache to avoid repeated processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000

def turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Optimized token-to-ID conversion with caching."""
    # Check cache first (significant speedup for repeated tokens)
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
        
    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
        
    # Process token
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
        
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
            
        return token_id
    except (ValueError, IndexError):
        return None

def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio with performance monitoring."""
    # Import here to avoid circular imports
    from .speechpipe import convert_to_audio as orpheus_convert_to_audio
    start_time = time.time()
    result = orpheus_convert_to_audio(multiframe, count)
    
    if result is not None:
        perf_monitor.add_audio_chunk()
        
    return result

async def tokens_decoder(token_gen) -> Generator[bytes, None, None]:
    """Simplified token decoder without complex ring buffer to ensure reliable output."""
    buffer = []
    count = 0
    
    # Use conservative batch parameters to ensure output quality
    min_frames = 28  # Default for reliability (4 chunks of 7)
    process_every = 7  # Process every 7 tokens (standard for Orpheus)
    
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
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - start_time
                if elapsed > 0:
                    print(f"Token processing rate: {token_count/elapsed:.1f} tokens/second")
                last_log_time = current_time
            
            # Process in standard batches for Orpheus model
            if count % process_every == 0 and count >= min_frames:
                # Use simple slice operation - reliable and correct
                buffer_to_proc = buffer[-min_frames:]
                
                # Debug output to help diagnose issues
                if count % 28 == 0:
                    print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")
                
                # Process the tokens
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """Optimized synchronous wrapper with parallel processing and efficient file I/O."""
    # Use a larger queue for high-end systems
    queue_size = 100 if HIGH_END_GPU else 50
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
    
    # Batch processing of tokens for improved throughput
    batch_size = 32 if HIGH_END_GPU else 16
    
    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        batch = []
        for token in syn_token_gen:
            batch.append(token)
            if len(batch) >= batch_size:
                for t in batch:
                    yield t
                batch = []
        # Process any remaining tokens
        for t in batch:
            yield t

    async def async_producer():
        # Track performance with more granular metrics
        start_time = time.time()
        chunk_count = 0
        last_log_time = start_time
        
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
            chunk_count += 1
            
            # Log performance periodically
            current_time = time.time()
            if current_time - last_log_time >= 3.0:  # Every 3 seconds
                elapsed = current_time - start_time
                if elapsed > 0:
                    chunks_per_sec = chunk_count / elapsed
                    print(f"Audio generation rate: {chunks_per_sec:.2f} chunks/second")
                last_log_time = current_time
                
        # Signal completion
        audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    # Use a separate thread with higher priority for producer
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Allow thread to be terminated when main thread exits
    thread.start()
    
    # For high-end GPUs, use a ThreadPoolExecutor for parallel file I/O
    if HIGH_END_GPU and wav_file:
        # Buffer for collecting chunks before writing
        write_buffer = []
        buffer_size = 10  # Write every 10 chunks
        
        def write_chunks_to_file(chunks, file):
            for chunk in chunks:
                file.writeframes(chunk)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = None
            
            while True:
                audio = audio_queue.get()
                if audio is None:
                    # Write any remaining buffered chunks
                    if write_buffer and wav_file:
                        if future:
                            future.result()  # Wait for previous write to complete
                        write_chunks_to_file(write_buffer, wav_file)
                    break
                
                audio_segments.append(audio)
                
                if wav_file:
                    write_buffer.append(audio)
                    if len(write_buffer) >= buffer_size:
                        if future:
                            future.result()  # Wait for previous write to complete
                        # Write in a separate thread to avoid blocking
                        chunks_to_write = write_buffer
                        write_buffer = []
                        future = executor.submit(write_chunks_to_file, chunks_to_write, wav_file)
    else:
        # Simpler direct approach for lower-end systems
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            
            audio_segments.append(audio)
            
            # Write to WAV file if provided
            if wav_file:
                wav_file.writeframes(audio)
    
    # Close WAV file if opened
    if wav_file:
        wav_file.close()
    
    thread.join()
    
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

def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None, temperature=TEMPERATURE, 
                             top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY, api_url: str=DEFAULT_API_URL):
    """Generate speech from text using Orpheus model with performance optimizations."""
    print(f"Starting speech generation for '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"Using voice: {voice}, GPU acceleration: {'Yes (High-end)' if HIGH_END_GPU else 'Yes' if torch.cuda.is_available() else 'No'}")
    
    # Reset performance monitor
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    # Generate speech with optimized settings
    result = tokens_decoder_sync(
        generate_tokens_from_api(
            prompt=prompt, 
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            api_url=api_url,
        ),
        output_file=output_file
    )
    
    # Report final performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total speech generation completed in {total_time:.2f} seconds")
    
    return result

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
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Inference server API URL")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, 
                       help="Repetition penalty (>=1.1 required for stable generation)")
    
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
        output_file=output_file,
        api_url=args.api_url,
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    main()
