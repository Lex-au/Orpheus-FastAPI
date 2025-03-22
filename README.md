![Orpheus-FASTAPI Banner](https://lex-au.github.io/Orpheus-FastAPI/Banner.png)

# Orpheus-FASTAPI

[![GitHub](https://img.shields.io/github/license/Lex-au/Orpheus-FastAPI)](https://github.com/Lex-au/Orpheus-FastAPI/blob/main/LICENSE.txt)

High-performance Text-to-Speech server with OpenAI-compatible API, 8 voices, emotion tags, and modern web UI. Optimized for RTX GPUs.

[GitHub Repository](https://github.com/Lex-au/Orpheus-FastAPI)

## Voice Demos

Listen to sample outputs with different voices and emotions:
- [Default Test Sample](https://lex-au.github.io/Orpheus-FastAPI/DefaultTest.mp3) - Standard neutral tone
- [Leah Happy Sample](https://lex-au.github.io/Orpheus-FastAPI/LeahHappy.mp3) - Cheerful, upbeat demo
- [Tara Sad Sample](https://lex-au.github.io/Orpheus-FastAPI/TaraSad.mp3) - Emotional, melancholic demo
- [Zac Contemplative Sample](https://lex-au.github.io/Orpheus-FastAPI/ZacContemplative.mp3) - Thoughtful, measured tone

## User Interface

![Web User Interface](https://lex-au.github.io/Orpheus-FastAPI/WebUI.png)

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- **Modern Web Interface**: Clean, responsive UI with waveform visualization
- **High Performance**: Optimized for RTX GPUs with parallel processing
- **Multiple Voices**: 8 different voice options with different characteristics
- **Emotion Tags**: Support for laughter, sighs, and other emotional expressions
- **Long-form Audio**: Efficient generation of extended audio content in a single request

## Project Structure

```
Orpheus-FastAPI/
├── app.py                # FastAPI server and endpoints
├── requirements.txt      # Dependencies
├── static/               # Static assets (favicon, etc.)
├── outputs/              # Generated audio files
├── templates/            # HTML templates
│   └── tts.html          # Web UI template
└── tts_engine/           # Core TTS functionality
    ├── __init__.py       # Package exports
    ├── inference.py      # Token generation and API handling
    └── speechpipe.py     # Audio conversion pipeline
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: RTX series for best performance)
- Separate LLM inference server running the Orpheus model (e.g., LM Studio or llama.cpp server)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Lex-au/Orpheus-FastAPI.git
cd Orpheus-FastAPI
```

2. Install `pipx`
```bash
sudo apt-get install pipx
```

3. Install `uv`
```bash
pipx install uv
```

4. Set up the required directories:
```bash
# Create directories for outputs and static files
mkdir -p outputs static
```

### Starting the Server

Run the FastAPI server:
```bash
uv run orpheus-fastapi
```

Or with specific host/port:
```bash
uv run orpheus-fastapi --host 0.0.0.0 --port 5005
```

![Terminal Output](https://lex-au.github.io/Orpheus-FastAPI/terminal.png)

Access:
- Web interface: http://localhost:5005/ (or http://127.0.0.1:5005/)
- API documentation: http://localhost:5005/docs (or http://127.0.0.1:5005/docs)

![API Documentation](https://lex-au.github.io/Orpheus-FastAPI/docs.png)

## API Usage

### OpenAI-Compatible Endpoint

The server provides an OpenAI-compatible API endpoint at `/v1/audio/speech`:

```bash
curl http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hello world! This is a test of the Orpheus TTS system.",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Parameters

- `input` (required): The text to convert to speech
- `model` (optional): The model to use (default: "orpheus")
- `voice` (optional): Which voice to use (default: "tara")
- `response_format` (optional): Output format (currently only "wav" is supported)
- `speed` (optional): Speed factor (0.5 to 1.5, default: 1.0)

### Legacy API

Additionally, a simpler `/speak` endpoint is available:

```bash
curl -X POST http://localhost:5005/speak \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world! This is a test.",
    "voice": "tara"
  }' \
  -o output.wav
```

### Available Voices

- `tara`: Female, conversational, clear
- `leah`: Female, warm, gentle
- `jess`: Female, energetic, youthful
- `leo`: Male, authoritative, deep
- `dan`: Male, friendly, casual
- `mia`: Female, professional, articulate
- `zac`: Male, enthusiastic, dynamic
- `zoe`: Female, calm, soothing

### Emotion Tags

You can insert emotion tags into your text to add expressiveness:

- `<laugh>`: Add laughter
- `<sigh>`: Add a sigh
- `<chuckle>`: Add a chuckle
- `<cough>`: Add a cough sound
- `<sniffle>`: Add a sniffle sound
- `<groan>`: Add a groan
- `<yawn>`: Add a yawning sound
- `<gasp>`: Add a gasping sound

Example: "Well, that's interesting <laugh> I hadn't thought of that before."

## Technical Details

This server works as a frontend that connects to an external LLM inference server. It sends text prompts to the inference server, which generates tokens that are then converted to audio using the SNAC model. The system has been optimised for RTX 4090 GPUs with:

- Vectorised tensor operations
- Parallel processing with CUDA streams
- Efficient memory management
- Token and audio caching
- Optimised batch sizes

For best performance, adjust the API_URL in `tts_engine/inference.py` to point to your LLM inference server endpoint.

### Integration with OpenWebUI

You can easily integrate this TTS solution with [OpenWebUI](https://github.com/open-webui/open-webui) to add high-quality voice capabilities to your chatbot:

1. Start your Orpheus-FASTAPI server
2. In OpenWebUI, go to Admin Panel > Settings > Audio
3. Change TTS from Web API to OpenAI
4. Set APIBASE URL to your server address (e.g., `http://localhost:5005`)
5. API Key can be set to "not-needed"
6. Set TTS Voice to one of the available voices: `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, or `zoe`
7. Set TTS Model to `tts-1`

### External Inference Server

This application requires a separate LLM inference server running the Orpheus model. You can use:

- [GPUStack](https://github.com/gpustack/gpustack) - GPU optimised LLM inference server (My pick) - supports LAN/WAN tensor split parallelisation
- [LM Studio](https://lmstudio.ai/) - Load the GGUF model and start the local server
- [llama.cpp server](https://github.com/ggerganov/llama.cpp) - Run with the appropriate model parameters
- Any compatible OpenAI API-compatible server

Download the quantised model from [lex-au/Orpheus-3b-FT-Q8_0.gguf](https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf) and load it in your inference server.

The inference server should be configured to expose an API endpoint that this FastAPI application will connect to.

### Environment Variables

You can configure the system by setting environment variables:

- `ORPHEUS_API_URL`: URL of the LLM inference API (tts_engine/inference.py)
- `ORPHEUS_API_TIMEOUT`: Timeout in seconds for API requests (default: 120)

Make sure the `ORPHEUS_API_URL` points to your running inference server.

## Development

### Project Components

- **app.py**: FastAPI server that handles HTTP requests and serves the web UI
- **tts_engine/inference.py**: Handles token generation and API communication 
- **tts_engine/speechpipe.py**: Converts token sequences to audio using the SNAC model

### Adding New Voices

To add new voices, update the `AVAILABLE_VOICES` list in `tts_engine/inference.py` and add corresponding descriptions in the HTML template.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE.txt file for details.
