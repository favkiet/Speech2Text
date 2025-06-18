from fastapi import FastAPI, UploadFile, HTTPException, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil
from pathlib import Path
import logging
import tempfile
from typing import Optional, List
from pydantic import BaseModel, Field
import json
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper.cpp Speech to Text API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and their content"""
    logger.info(f"Request: {request.method} {request.url}")
    
    # Log request headers
    logger.info("Headers:")
    for name, value in request.headers.items():
        logger.info(f"  {name}: {value}")
    
    # Try to log request body for debugging
    try:
        body = await request.body()
        logger.info(f"Request body length: {len(body)} bytes")
        if len(body) < 1000:  # Don't log large bodies
            logger.info(f"Request body: {body.decode()}")
    except Exception as e:
        logger.warning(f"Could not log request body: {e}")
    
    response = await call_next(request)
    return response

class WhisperParams(BaseModel):
    """Parameters for Whisper transcription"""
    # Threading and processing
    n_threads: Optional[int] = Field(5, description="Number of threads to use during computation")
    n_processors: Optional[int] = Field(1, description="Number of processors to use during computation")
    
    # Beam search parameters
    best_of: Optional[int] = Field(5, description="Number of best candidates to keep")
    beam_size: Optional[int] = Field(5, description="Beam size for beam search")
    
    # Audio processing parameters
    word_thold: Optional[float] = Field(0.01, description="Word timestamp probability threshold")
    entropy_thold: Optional[float] = Field(2.40, description="Entropy threshold for decoder fail")
    logprob_thold: Optional[float] = Field(-1.00, description="Log probability threshold for decoder fail")
    no_speech_thold: Optional[float] = Field(0.60, description="No speech threshold")
    
    # Temperature settings
    temperature: Optional[float] = Field(0.0, description="Temperature for sampling")
    temperature_inc: Optional[float] = Field(0.2, description="Temperature increment")
    
    # Audio context
    audio_ctx: Optional[int] = Field(0, description="Audio context size (0 = use defaults)")
    max_context: Optional[int] = Field(-1, description="Maximum number of text context tokens to store")
    max_len: Optional[int] = Field(0, description="Maximum segment length in characters")
    
    # Processing flags
    split_on_word: Optional[bool] = Field(False, description="Split on word rather than on token")
    no_fallback: Optional[bool] = Field(False, description="Do not use temperature fallback while decoding")
    
    # Output format flags
    output_txt: Optional[bool] = Field(True, description="Output result in a text file")
    output_vtt: Optional[bool] = Field(False, description="Output result in a VTT file")
    output_srt: Optional[bool] = Field(False, description="Output result in a SRT file")
    output_lrc: Optional[bool] = Field(False, description="Output result in a LRC file")
    output_wts: Optional[bool] = Field(False, description="Output result in a word timestamps file")
    output_csv: Optional[bool] = Field(False, description="Output result in a CSV file")
    output_json: Optional[bool] = Field(False, description="Output result in a JSON file")
    output_json_full: Optional[bool] = Field(False, description="Output result in a full JSON file")
    
    # Display flags
    no_prints: Optional[bool] = Field(True, description="Do not print any text output")
    print_special: Optional[bool] = Field(False, description="Print special tokens")
    print_colors: Optional[bool] = Field(False, description="Print colors")
    print_progress: Optional[bool] = Field(False, description="Print progress")
    no_timestamps: Optional[bool] = Field(False, description="Do not print timestamps")
    
    # Language and model
    language: Optional[str] = Field("vi", description="Language code (e.g., 'en', 'vi', etc.)")
    detect_language: Optional[bool] = Field(False, description="Detect language automatically")
    prompt: Optional[str] = Field(None, description="Initial prompt for the model")
    
    # GPU and performance
    use_gpu: Optional[bool] = Field(True, description="Use GPU for inference")
    flash_attn: Optional[bool] = Field(False, description="Use flash attention")
    
    # Other parameters
    suppress_nst: Optional[bool] = Field(False, description="Suppress non-speech tokens")
    suppress_regex: Optional[str] = Field(None, description="Regular expression to suppress")
    grammar: Optional[str] = Field(None, description="Grammar for the model")
    grammar_rule: Optional[str] = Field(None, description="Grammar rule to use")
    grammar_penalty: Optional[float] = Field(None, description="Grammar penalty")

# Load environment variables from .env file
load_dotenv()

# Get base paths from environment or use defaults relative to current file
BASE_DIR = Path(__file__).resolve().parent
WHISPER_PATH = os.getenv('WHISPER_PATH', str(BASE_DIR / "whisper.cpp"))
WHISPER_CLI = os.getenv('WHISPER_CLI', str(Path(WHISPER_PATH) / "build/bin/whisper-cli"))
MODEL_PATH = os.getenv('MODEL_PATH', str(Path(WHISPER_PATH) / "models/ggml-large-v3-turbo.bin"))

# Create directories for processing
UPLOAD_DIR = os.getenv('UPLOAD_DIR', str(BASE_DIR / "uploads"))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', str(BASE_DIR / "output"))
TEMP_DIR = tempfile.mkdtemp()
TEMP_UPLOAD_DIR = os.path.join(TEMP_DIR, "uploads")

# Create all necessary directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_UPLOAD_DIR]:
    os.makedirs(directory, exist_ok=True)

# Validate required paths
def validate_paths():
    """Validate that all required paths exist"""
    if not os.path.exists(WHISPER_CLI):
        raise RuntimeError(
            f"whisper-cli not found at {WHISPER_CLI}. "
            "Please set WHISPER_CLI environment variable or build whisper.cpp"
        )
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Please set MODEL_PATH environment variable or download the model"
        )

# Validate paths on startup
validate_paths()

def run_whisper_cli(command: List[str]) -> tuple[bool, str, str]:
    """Run whisper-cli command and return success status and output"""
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        # Log the complete output
        if process.stdout:
            logger.info(f"Whisper CLI stdout:\n{process.stdout}")
        if process.stderr:
            logger.info(f"Whisper CLI stderr:\n{process.stderr}")
            
        return process.returncode == 0, process.stdout, process.stderr
    except Exception as e:
        logger.error(f"Error running whisper-cli: {str(e)}")
        return False, "", str(e)

def build_whisper_command(params: WhisperParams, input_file: str, output_file: str) -> List[str]:
    """Build whisper-cli command with all necessary parameters"""
    cmd = [
        WHISPER_CLI,
        "-m", MODEL_PATH,
        "-f", input_file,
        "-of", output_file
    ]

    # Add all parameters
    if params.n_threads: cmd.extend(["-t", str(params.n_threads)])
    if params.n_processors: cmd.extend(["-p", str(params.n_processors)])
    if params.best_of: cmd.extend(["-bo", str(params.best_of)])
    if params.beam_size: cmd.extend(["-bs", str(params.beam_size)])
    if params.word_thold: cmd.extend(["-wt", str(params.word_thold)])
    if params.entropy_thold: cmd.extend(["-et", str(params.entropy_thold)])
    if params.logprob_thold: cmd.extend(["-lpt", str(params.logprob_thold)])
    if params.no_speech_thold: cmd.extend(["-nth", str(params.no_speech_thold)])
    if params.temperature: cmd.extend(["-tp", str(params.temperature)])
    if params.temperature_inc: cmd.extend(["-tpi", str(params.temperature_inc)])
    if params.audio_ctx: cmd.extend(["-ac", str(params.audio_ctx)])
    if params.max_context: cmd.extend(["-mc", str(params.max_context)])
    if params.max_len: cmd.extend(["-ml", str(params.max_len)])
    
    # Add boolean flags
    if params.split_on_word: cmd.append("-sow")
    if params.no_fallback: cmd.append("-nf")
    if params.output_txt: cmd.append("-otxt")
    if params.output_vtt: cmd.append("-ovtt")
    if params.output_srt: cmd.append("-osrt")
    if params.output_lrc: cmd.append("-olrc")
    if params.output_wts: cmd.append("-owts")
    if params.output_csv: cmd.append("-ocsv")
    if params.output_json: cmd.append("-oj")
    if params.output_json_full: cmd.append("-ojf")
    if params.no_prints: cmd.append("-np")
    if params.print_special: cmd.append("-ps")
    if params.print_colors: cmd.append("-pc")
    if params.print_progress: cmd.append("-pp")
    if params.no_timestamps: cmd.append("-nt")
    if params.detect_language: cmd.append("-dl")
    if params.suppress_nst: cmd.append("-sns")
    
    # Add language
    if params.language: cmd.extend(["-l", params.language])
    
    # Add optional string parameters
    if params.prompt: cmd.extend(["--prompt", params.prompt])
    if params.suppress_regex: cmd.extend(["--suppress-regex", params.suppress_regex])
    if params.grammar: cmd.extend(["--grammar", params.grammar])
    if params.grammar_rule: cmd.extend(["--grammar-rule", params.grammar_rule])
    if params.grammar_penalty: cmd.extend(["--grammar-penalty", str(params.grammar_penalty)])
    
    # GPU settings
    if not params.use_gpu: cmd.append("-ng")
    if params.flash_attn: cmd.append("-fa")
    
    return cmd

def cleanup_temp_files(*files):
    """Clean up temporary files."""
    for file in files:
        try:
            if os.path.exists(file) and TEMP_DIR in file:  # Only delete temp files
                os.remove(file)
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {file}: {str(e)}")

def find_output_file(base_path: str) -> str:
    """Find the actual output file, handling potential .txt.txt issue"""
    # Try potential file paths
    candidates = [
        base_path,           # original path
        base_path + '.txt',  # with extra .txt
        base_path[:-4]       # without .txt
    ]
    
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Found output file at: {path}")
            return path
            
    logger.error(f"No output file found. Tried paths: {candidates}")
    return None

def save_transcription(temp_output: str, filename: str) -> str:
    """Save transcription to permanent storage"""
    # Get base name without extension and add .txt
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}.txt"
    
    # Create permanent output path
    permanent_output = os.path.join(OUTPUT_DIR, output_filename)
    
    # Copy the file
    try:
        shutil.copy2(temp_output, permanent_output)
        logger.info(f"Saved transcription to: {permanent_output}")
        return permanent_output
    except Exception as e:
        logger.error(f"Failed to save transcription: {str(e)}")
        raise

@app.post("/transcribe/")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    params: Optional[str] = Form(None)
):
    """
    Transcribe audio file using whisper.cpp
    """
    try:
        # Log received file info
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Content type: {file.content_type}")

        # Parse parameters
        params_obj = WhisperParams()
        if params:
            try:
                params_dict = json.loads(params)
                params_obj = WhisperParams(**params_dict)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in params")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
        
        logger.info(f"Received params: {params_obj.dict(exclude_none=True)}")

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create uploads directory
            upload_dir = os.path.join(temp_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save uploaded file
            temp_input = os.path.join(upload_dir, file.filename)
            with open(temp_input, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved temp file to: {temp_input}")
            
            # Prepare output path
            temp_output = os.path.join(temp_dir, os.path.splitext(file.filename)[0] + ".txt")
            
            # Build and run whisper command
            cmd = build_whisper_command(params_obj, temp_input, temp_output)
            logger.info(f"Running command: {' '.join(cmd)}")
            success, stdout, stderr = run_whisper_cli(cmd)
            
            if not success:
                logger.error(f"Transcription failed: {stderr}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {stderr}")
            
            # Find the output file (it might have .txt.txt extension)
            actual_output = find_output_file(temp_output)
            if not actual_output or not os.path.exists(actual_output):
                raise HTTPException(status_code=500, detail="Output file not found")
            
            logger.info(f"Found output file at: {actual_output}")
            
            # Save to permanent storage
            permanent_output = save_transcription(actual_output, file.filename)
            logger.info(f"Saved transcription to: {permanent_output}")
            
            # Read transcription
            try:
                with open(actual_output, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
                logger.info("Successfully read transcription")
            except Exception as e:
                logger.error(f"Error reading transcription: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error reading transcription: {str(e)}")
            
            return {
                "success": True,
                "language": params_obj.language,
                "detect_language": params_obj.detect_language,
                "transcription": transcription,
                "output_file": permanent_output
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Cleanup is handled by tempfile.TemporaryDirectory context manager
        pass

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Whisper.cpp Speech to Text API is running",
        "model": os.path.basename(MODEL_PATH),
        "supported_tasks": ["transcribe", "translate"],
        "default_language": "vi"
    }

# Cleanup temp directory on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    try:
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 