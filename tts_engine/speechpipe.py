from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys

# Helper to detect if running in Uvicorn's reloader (same as in inference.py)
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

# Set a flag to avoid repeat messages
IS_RELOADER = is_reloader_process()

# Detect hardware capabilities
APPLE_SILICON = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

# Set device for model processing
if APPLE_SILICON:
    DEVICE = "mps"
    if not IS_RELOADER:
        print("🍎 Using Apple Silicon MPS for speech generation")
elif CUDA_AVAILABLE:
    DEVICE = "cuda"
    if not IS_RELOADER:
        print("🖥️ Using CUDA for speech generation")
else:
    DEVICE = "cpu"
    if not IS_RELOADER:
        print("⚙️ Using CPU for speech generation")

# Check if CoreML should be enabled
# USE_COREML = APPLE_SILICON and os.environ.get("ORPHEUS_USE_COREML", "1") == "1"
# CoreML logic removed

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile') and not APPLE_SILICON:  # torch.compile not fully supported on MPS yet
        TORCH_COMPILE_AVAILABLE = True
        if not IS_RELOADER:
            print("PyTorch 2.0+ detected, torch.compile is available")
except:
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if CUDA_AVAILABLE and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        if not IS_RELOADER:
            print("CUDA graphs support is available")
except:
    pass

# Load the model with appropriate device placement
base_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
base_model = base_model.to(DEVICE)

# Assign base_model directly as the model to use
model = base_model

# Check if CoreML wrapper should be used
# if USE_COREML:
#     try:
#         from .coreml_wrapper import CoreMLWrapper
#         # Wrap the base model with CoreML for Apple Silicon
#         model = CoreMLWrapper(base_model, device=DEVICE)
#         if model.use_coreml and model.coreml_model is not None:
#             if not IS_RELOADER:
#                 print("🧠 Using CoreML Neural Engine acceleration!")
#         else:
#             if not IS_RELOADER:
#                 print("CoreML model not available, using MPS acceleration instead")
#     except ImportError:
#         # If CoreML wrapper is not available, use the base model
#         model = base_model
#         if not IS_RELOADER:
#             print("CoreML wrapper not available, using standard PyTorch backend")
# else:
#     # Use base model directly
#     model = base_model

if not IS_RELOADER:
    print(f"SNAC model loaded directly on {DEVICE} (CoreML export disabled)")

# Disable torch.compile for MPS as it's not fully supported
if APPLE_SILICON:
    if not IS_RELOADER:
        print("Using standard PyTorch optimizations for Apple Silicon")
elif TORCH_COMPILE_AVAILABLE:
    if not IS_RELOADER:
        print("Using torch.compile for optimized performance")
else:
    if not IS_RELOADER:
        print("Using standard PyTorch optimizations")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if CUDA_AVAILABLE:
    cuda_stream = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using CUDA stream for parallel processing")


def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio that supports Apple Silicon MPS and
    eliminates inefficient tensor operations for much faster inference.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=DEVICE)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=DEVICE)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=DEVICE)
    
    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=DEVICE)
    
    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7
        
        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]
        
        # Code 1 - two values per frame
        codes_1[j*2] = frame_tensor[idx+1]
        codes_1[j*2+1] = frame_tensor[idx+4]
        
        # Code 2 - four values per frame
        codes_2[j*4] = frame_tensor[idx+2]
        codes_2[j*4+1] = frame_tensor[idx+3]
        codes_2[j*4+2] = frame_tensor[idx+5]
        codes_2[j*4+3] = frame_tensor[idx+6]
    
    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0), 
        codes_1.unsqueeze(0), 
        codes_2.unsqueeze(0)
    ]
    
    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Context manager depends on device type
    if CUDA_AVAILABLE:
        stream_ctx = torch.cuda.stream(cuda_stream)
    else:
        stream_ctx = torch.inference_mode()
    
    with stream_ctx:
        # Decode the audio using the base model directly
        audio_hat = model.decode(codes) # model is now directly base_model
        
        # Extract the relevant slice and efficiently convert to bytes
        # Keep data on GPU as long as possible
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Process based on device type to minimize data transfers
        if CUDA_AVAILABLE:
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        elif APPLE_SILICON:
            # For MPS, we need to go through CPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            audio_bytes = audio_int16_tensor.detach().cpu().numpy().tobytes()
        else:
            # For CPU, simpler pathway
            audio_np = audio_slice.detach().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
    return audio_bytes

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000  # Increased cache size for better performance

def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with caching.
    This is the definitive implementation used by both inference.py and speechpipe.py.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        int: Token ID if valid, None otherwise
    """
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

async def tokens_decoder(token_gen):
    """Optimized token decoder with early first-chunk processing for lower latency"""
    buffer = []
    count = 0

    # Track if first chunk has been processed
    first_chunk_processed = False
    
    # Use different thresholds for first chunk vs. subsequent chunks
    min_frames_first = 7  # Just one chunk (7 tokens) for first audio - ultra-low latency
    min_frames_subsequent = 28  # Standard minimum (4 chunks of 7 tokens) after first audio
    ideal_frames = 49  # Ideal standard frame size (7×7 window) - unchanged
    process_every_n = 7  # Process every 7 tokens (standard for Orpheus model) - unchanged
    
    start_time = time.time()
    token_count = 0
    last_log_time = start_time
    
    async for token_sim in token_gen:
        token_count += 1
        
        # Use the unified turn_token_into_id which already handles caching
        token = turn_token_into_id(token_sim, count)
        
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Log throughput periodically
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    recent_tokens = token_count
                    tokens_per_sec = recent_tokens / elapsed
                    print(f"Token processing rate: {tokens_per_sec:.1f} tokens/second")
                last_log_time = current_time
                token_count = 0
            
            # Different processing logic based on whether first chunk has been processed
            if not first_chunk_processed:
                # Process first chunk as soon as possible for minimal latency
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    
                    # Process the first chunk of audio for immediate feedback
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens for low latency")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True  # Mark first chunk as processed
                        yield audio_samples
            else:
                # For subsequent chunks, use original processing with proper batching
                if count % process_every_n == 0:
                    # Use same prioritization logic as before
                    if len(buffer) >= ideal_frames:
                        buffer_to_proc = buffer[-ideal_frames:]
                    elif len(buffer) >= min_frames_subsequent:
                        buffer_to_proc = buffer[-min_frames_subsequent:]
                    else:
                        continue
                    
                    # Debug output to help diagnose issues
                    if count % 28 == 0:
                        print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")
                    
                    # Process the tokens
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
    
    # CRITICAL: End-of-generation handling - process all remaining frames
    # Process remaining complete frames (ideal size)
    if len(buffer) >= ideal_frames:
        buffer_to_proc = buffer[-ideal_frames:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
            
    # Process any additional complete frames (minimum size)
    elif len(buffer) >= min_frames_subsequent:
        buffer_to_proc = buffer[-min_frames_subsequent:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
            
    # Final special case: even if we don't have minimum frames, try to process
    # what we have by padding with silence tokens that won't affect the audio
    elif len(buffer) >= process_every_n:
        # Pad to minimum frame requirement with copies of the final token
        # This is more continuous than using unrelated tokens from the beginning
        last_token = buffer[-1]
        padding_needed = min_frames_subsequent - len(buffer)
        
        # Create a padding array of copies of the last token
        # This maintains continuity much better than circular buffering
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding
        
        print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} repeated-token padding")
        audio_samples = convert_to_audio(padded_buffer, count)
        if audio_samples is not None:
            yield audio_samples

def reset_state():
    """Reset all state between batch processing."""
    global token_id_cache, cuda_stream
    
    # Clear the token ID cache
    token_id_cache.clear()
    
    # Reset CUDA stream if available
    if CUDA_AVAILABLE and cuda_stream is not None:
        cuda_stream.synchronize()
        cuda_stream = torch.cuda.Stream()
        
        # Clear CUDA cache 
        torch.cuda.empty_cache()
    
    # For Apple Silicon, clear MPS cache if available
    if APPLE_SILICON:
        try:
            # This is the proper way to clear MPS cache in PyTorch 2.0+
            torch.mps.empty_cache()
        except:
            # Fallback for older PyTorch versions or if the operation fails
            pass
            
    # Reset CoreML state if needed
    # if USE_COREML and isinstance(model, CoreMLWrapper) and model.use_coreml:
    #     try:
    #         # CoreML resources are generally managed by the OS
    #         # but we'll manually trigger garbage collection to be safe
    #         import gc
    #         gc.collect()
    #     except:
    #         pass
    
    # Force garbage collection
    import gc
    gc.collect()

# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with hardware-specific optimizations"""
    # Set queue size based on hardware
    if APPLE_SILICON:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 64:  # High-memory Apple Silicon
            max_queue_size = 64
            batch_size = 32
        elif ram_gb >= 32:  # Mid-range Apple Silicon
            max_queue_size = 48
            batch_size = 24
        else:  # Base model
            max_queue_size = 32
            batch_size = 16
    elif CUDA_AVAILABLE:
        max_queue_size = 32
        batch_size = 16
    else:  # CPU fallback
        max_queue_size = 8
        batch_size = 4
    
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        token_batch = []
        for token in syn_token_gen:
            token_batch.append(token)
            # Process in batches for efficiency
            if len(token_batch) >= batch_size:
                for t in token_batch:
                    yield t
                token_batch = []
        # Process any remaining tokens
        for t in token_batch:
            yield t

    async def async_producer():
        # Start timer for performance logging
        start_time = time.time()
        chunk_count = 0
        
        try:
            # Process audio chunks from the token decoder
            async for audio_chunk in tokens_decoder(async_token_gen()):
                if audio_chunk:  # Validate audio chunk before adding to queue
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    # Log performance stats periodically
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Generated {chunk_count} chunks in {elapsed:.2f}s ({chunk_count/elapsed:.2f} chunks/sec)")
        except Exception as e:
            print(f"Error in audio producer: {e}")
            import traceback
            traceback.print_exc()
        finally:    
            # Signal completion
            print("Audio producer completed - finalizing all chunks")
            audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    # Create thread with appropriate priority
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    thread.start()

    # Use hardware-specific buffer sizes
    if APPLE_SILICON:
        buffer_size = 8  # Larger buffer for smoother playback on Apple Silicon
    else:
        buffer_size = 5
    audio_buffer = []
    
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_buffer.append(audio)
        # Yield buffered audio chunks for smoother playback
        if len(audio_buffer) >= buffer_size:
            for chunk in audio_buffer:
                yield chunk
            audio_buffer = []
    
    # Yield any remaining audio in the buffer
    for chunk in audio_buffer:
        yield chunk

    thread.join()
