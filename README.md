# Speech to Text API using Whisper.cpp

FastAPI application for speech-to-text transcription using whisper.cpp. This application supports Vietnamese language transcription with high accuracy.

## Prerequisites

- Python 3.10+
- FFmpeg
- Git
- Build tools (gcc, cmake, etc.)
- pip or Conda package manager

## Installation and Setup

You can choose either pip or Conda for installation:

### Option 1: Using pip

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Speech2Text.git
cd Speech2Text
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Speech2Text.git
cd Speech2Text
```

2. Create and activate Conda environment:
```bash
conda create -n stt python=3.10
conda activate stt
```

3. Install dependencies:
```bash
# Install pip dependencies
pip install -r requirements.txt

# Install FFmpeg using Conda
conda install -c conda-forge ffmpeg
```

### Common Setup Steps (After pip or Conda installation)

4. Clone and build whisper.cpp:
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
```

5. Download the model:
```bash
cd whisper.cpp/models
bash download-ggml-model.sh [model-name]
```

Available models:

| Model | Size | Speed | Use Case | Description |
|-------|------|-------|----------|-------------|
| tiny | 75M | ⚡️⚡️⚡️ | Quick testing | Fastest, lowest accuracy |
| tiny.en | 75M | ⚡️⚡️⚡️ | English only | Optimized for English |
| base | 142M | ⚡️⚡️ | Simple tasks | Good balance for basic use |
| base.en | 142M | ⚡️⚡️ | English only | Optimized for English |
| small | 466M | ⚡️⚡️ | Development | Good balance of speed/accuracy |
| small.en | 466M | ⚡️⚡️ | English only | Optimized for English |
| small.en-tdrz | 465M | ⚡️⚡️ | Speaker detection | Supports local diarization |
| medium | 1.5G | ⚡️ | Better accuracy | Recommended for multilingual |
| medium.en | 1.5G | ⚡️ | English only | Optimized for English |
| large-v1 | 2.9G | 🐢 | Legacy | Original large model |
| large-v2 | 2.9G | 🐢 | Legacy | Improved large model |
| large-v2-q5_0 | 1.1G | ⚡️ | Quantized | Compressed v2, faster |
| large-v3 | 2.9G | 🐢 | Best accuracy | Latest version |
| large-v3-q5_0 | 1.1G | ⚡️ | Quantized | Compressed v3, faster |
| large-v3-turbo | 1.5G | ⚡️ | Production | Best speed/accuracy balance |
| large-v3-turbo-q5_0 | 547M | ⚡️⚡️ | Quantized | Compressed turbo, fastest large |

Notes:
- Models with `.en` suffix are English-only models
- Models with `q5_0` suffix are quantized (compressed) versions
- Models with `tdrz` suffix support speaker detection
- Quantized models offer faster inference with slightly lower accuracy
- Size refers to the model file size on disk

Recommendations:
- Development/Testing: Use `small` or `medium` model
- Production/High Accuracy: Use `large-v3-turbo` model
- Limited Resources: Use `base` model
- English Only: Use corresponding `.en` models for better performance
- Low Memory Systems: Use quantized (`q5_0`) versions

6. Create .env file:
```bash
WHISPER_PATH=/path/to/whisper.cpp
WHISPER_CLI=/path/to/whisper.cpp/build/bin/whisper-cli
MODEL_PATH=/path/to/whisper.cpp/models/ggml-large-v3-turbo.bin  # Change based on chosen model
UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
```

7. Create required directories:
```bash
mkdir uploads output
```

8. Run the FastAPI application:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Transcribe Audio (`POST /transcribe/`)

Endpoint for transcribing audio files.

**Request:**
- Method: `POST`
- URL: `http://localhost:8000/transcribe/`
- Content-Type: `multipart/form-data`
- Body Parameters:
  - `file`: Audio file (mp3, wav, etc.)
  - `params` (optional): JSON string with transcription parameters
    ```json
    {
        "language": "vi",
        "n_threads": 5,
        "best_of": 5,
        "beam_size": 5
    }
    ```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/transcribe/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-audio-file.mp3" \
  -F 'params={"language": "vi"}'
```

**Example using Postman:**
1. Create a new POST request to `http://localhost:8000/transcribe/`
2. In the Body tab:
   - Select `form-data`
   - Add key `file` (Type: File) and select your audio file
   - Add key `params` (Type: Text) and enter the JSON parameters

### 2. Health Check (`GET /`)

Check if the API is running.

```bash
curl http://localhost:8000/
```

## Project Structure

```
Speech2Text/
├── app.py              # FastAPI application
├── requirements.txt    # Python dependencies
├── uploads/           # Directory for uploaded files
├── output/            # Directory for transcription output
└── whisper.cpp/       # Whisper.cpp library
```

## Environment Variables

- `WHISPER_PATH`: Path to whisper.cpp directory
- `WHISPER_CLI`: Path to whisper-cli executable
- `MODEL_PATH`: Path to the model file
- `UPLOAD_DIR`: Directory for uploaded files
- `OUTPUT_DIR`: Directory for transcription output

## Notes

- The application uses the specified model for transcription
- Transcription results are saved in both temporary and permanent storage
- Supports various audio formats (mp3, wav, etc.)
- Optimized for Vietnamese language transcription 