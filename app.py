from fastapi import FastAPI, UploadFile, HTTPException, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
from pathlib import Path
import logging
import shutil
from typing import Optional
import json
import time
import tempfile
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('log/app.log')
    ]
)
logger = logging.getLogger(__name__)

# Get base paths from environment or use defaults relative to current file
BASE_DIR = Path(__file__).resolve().parent
WHISPER_PATH = os.getenv('WHISPER_PATH', str(BASE_DIR / "whisper.cpp"))
WHISPER_CLI = os.getenv('WHISPER_CLI', str(Path(WHISPER_PATH) / "build/bin/whisper-cli"))
MODEL_PATH = os.getenv('MODEL_PATH', str(Path(WHISPER_PATH) / "models/ggml-large-v3-turbo.bin"))

# Create directories for processing
OUTPUT_DIR = os.getenv('OUTPUT_DIR', str(BASE_DIR / "output"))
TEMP_DIR = tempfile.mkdtemp()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate paths
if not os.path.exists(WHISPER_PATH):
    raise RuntimeError(f"Whisper directory not found at: {WHISPER_PATH}")
if not os.path.exists(WHISPER_CLI):
    raise RuntimeError(f"whisper-cli not found at: {WHISPER_CLI}")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at: {MODEL_PATH}")

# logger.info(f"Using Speech2Text directory: {BASE_DIR}")
# logger.info(f"Using whisper.cpp directory: {WHISPER_PATH}")
# logger.info(f"Using whisper-cli: {WHISPER_CLI}")
# logger.info(f"Using model: {MODEL_PATH}")
# logger.info(f"Using output directory: {OUTPUT_DIR}")
# logger.info(f"Using temp directory: {TEMP_DIR}")

class WhisperParams(BaseModel):
    """Parameters for Whisper transcription"""
    # Threading and processing
    n_threads: Optional[int] = Field(4, description="Number of threads to use")
    language: Optional[str] = Field("vi", description="Language code (e.g., 'vi')")
    output_txt: Optional[bool] = Field(True, description="Output result in a text file")
    no_prints: Optional[bool] = Field(True, description="Do not print any text output")
    use_gpu: Optional[bool] = Field(False, description="Use GPU for inference")

app = FastAPI(title="Whisper.cpp Speech to Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_whisper_command(input_file: str, output_file: str, params: WhisperParams) -> tuple[bool, str]:
    """Run whisper-cli command"""
    try:
        # Store current directory
        original_dir = os.getcwd()
        logger.info(f"Current directory before change: {original_dir}")
        
        # Change to whisper.cpp directory
        whisper_dir = os.path.abspath(WHISPER_PATH)
        os.chdir(whisper_dir)
        logger.info(f"Changed working directory to: {whisper_dir}")
        
        # Build command with absolute paths
        cmd = [
            WHISPER_CLI,
            "-f", os.path.abspath(input_file),
            "-m", MODEL_PATH,
            "-otxt",  # Let whisper-cli handle the output path
            "-l", params.language,
            "-t", str(params.n_threads)
        ]
        
        if not params.use_gpu:
            cmd.append("-ng")
        
        # Log command details
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("Command components:")
        for i, part in enumerate(cmd):
            logger.info(f"  {i}: {part}")
        
        # Run command without capturing output to see realtime progress
        logger.info("Starting subprocess.run...")
        result = subprocess.run(
            cmd,
            capture_output=False,  # Don't capture output
            text=True,
            timeout=1200
        )
        logger.info("subprocess.run completed")
        logger.info(f"Process return code: {result.returncode}")
        
        # Change back to original directory
        os.chdir(original_dir)
        logger.info(f"Changed back to directory: {original_dir}")
        
        # Get actual output path from whisper-cli output
        actual_output = input_file + ".txt"
        
        return result.returncode == 0, actual_output
        
    except subprocess.TimeoutExpired:
        logger.error("Whisper process timed out after 10 minutes")
        try:
            os.chdir(original_dir)
        except:
            pass
        return False, ""
    except Exception as e:
        logger.error(f"Error running whisper command: {str(e)}", exc_info=True)
        try:
            os.chdir(original_dir)
        except:
            pass
        return False, ""

@app.post("/transcribe/")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    params: Optional[str] = Form(None)
):
    """
    Transcribe audio file using whisper.cpp
    """
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    try:
        # Initial debug point
        logger.info(f"[{request_id}] Starting transcribe request...")
        
        # Check if file is None
        if not file:
            logger.error(f"[{request_id}] No file received")
            raise HTTPException(status_code=400, detail="No file received")
            
        # Check if file has filename
        if not file.filename:
            logger.error(f"[{request_id}] No filename")
            raise HTTPException(status_code=400, detail="No filename")
            
        # Log received file info
        logger.info(f"[{request_id}] Received file: {file.filename}")
        logger.info(f"[{request_id}] Content type: {file.content_type}")
        
        # Try to get file size
        try:
            contents = await file.read()
            size = len(contents)
            logger.info(f"[{request_id}] File size: {size} bytes")
            if size == 0:
                logger.error(f"[{request_id}] Empty file received")
                raise HTTPException(status_code=400, detail="Empty file received")
        except Exception as e:
            logger.error(f"[{request_id}] Error getting file size: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

        # Parse parameters
        logger.info(f"[{request_id}] Starting parameter parsing...")
        params_obj = WhisperParams()
        if params:
            try:
                logger.info(f"[{request_id}] Raw params: {params}")
                params_dict = json.loads(params)
                logger.info(f"[{request_id}] Parsed params dict: {params_dict}")
                params_obj = WhisperParams(**params_dict)
                logger.info(f"[{request_id}] Parameters parsed successfully: {params_obj.dict(exclude_none=True)}")
            except json.JSONDecodeError as e:
                logger.error(f"[{request_id}] Invalid JSON in params: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON in params: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Invalid parameters: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
        
        # Create temporary file path
        temp_input = os.path.join(TEMP_DIR, file.filename)
        
        # Save uploaded file to temp directory
        logger.info(f"[{request_id}] Saving file to: {temp_input}")
        try:
            with open(temp_input, "wb") as buffer:
                buffer.write(contents)
            logger.info(f"[{request_id}] File saved successfully. Total size: {size/1024/1024:.1f}MB")
        except Exception as e:
            logger.error(f"[{request_id}] Error saving file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        # Run whisper command
        logger.info(f"[{request_id}] Starting whisper transcription...")
        success, actual_output = run_whisper_command(
            input_file=temp_input,
            output_file="",  # Not used anymore
            params=params_obj
        )
        
        if not success:
            logger.error(f"[{request_id}] Transcription failed")
            raise HTTPException(status_code=500, detail="Transcription failed")
            
        # Check if output file exists
        if not os.path.exists(actual_output):
            logger.error(f"[{request_id}] Output file not found at: {actual_output}")
            raise HTTPException(status_code=500, detail="Output file not found")
            
        # Copy result to output directory
        output_filename = os.path.splitext(file.filename)[0] + ".txt"
        permanent_output = os.path.join(OUTPUT_DIR, output_filename)
        
        shutil.copy2(actual_output, permanent_output)
        logger.info(f"[{request_id}] Saved transcription to: {permanent_output}")
        
        # Read transcription
        try:
            logger.info(f"[{request_id}] Reading transcription...")
            with open(permanent_output, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
            logger.info(f"[{request_id}] Successfully read transcription")
        except Exception as e:
            logger.error(f"[{request_id}] Error reading transcription: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading transcription: {str(e)}")
            
        # Cleanup temp files
        try:
            os.remove(temp_input)
            os.remove(actual_output)
            logger.info(f"[{request_id}] Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"[{request_id}] Error cleaning up temp files: {str(e)}")
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"[{request_id}] Total processing time: {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "language": params_obj.language,
            "transcription": transcription,
            "output_file": permanent_output,
            "processing_time": f"{processing_time:.2f}s"
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Whisper.cpp Speech to Text API is running"
    }

# Cleanup temp directory on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    try:
        shutil.rmtree(TEMP_DIR)
        logger.info("Cleaned up temp directory on shutdown")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 