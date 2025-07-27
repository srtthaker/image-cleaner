import io
import logging
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --- Model Imports ---
from rembg import remove as remove_bg_rembg
from simple_lama_inpainting import SimpleLama

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Combined AI Photo Editor API",
    description="A single API for both background removal and inpainting.",
    version="2.0.0",
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
lama_model = None
try:
    logging.info("Attempting to load LaMa Inpainting model...")
    lama_model = SimpleLama(device="cpu")
    logging.info("LaMa Inpainting model loaded successfully on CPU.")
except Exception as e:
    logging.error(f"Fatal error loading LaMa model: {e}")
    logging.error(traceback.format_exc())

# --- API Endpoints ---
@app.get("/")
def read_root():
    """ Root endpoint to check if the API is running. """
    return {"message": "Welcome to the Combined AI Photo Editor API!"}


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    """
    Endpoint to remove the background from an uploaded image using rembg.
    """
    logging.info(f"Received request for /remove-bg for file: {file.filename}")
    try:
        input_image_bytes = await file.read()
        output_image_bytes = remove_bg_rembg(input_image_bytes)
        return StreamingResponse(io.BytesIO(output_image_bytes), media_type="image/png")
    except Exception as e:
        logging.error(f"An error occurred during background removal: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/inpaint")
async def inpaint_image(file: UploadFile = File(...), mask_file: UploadFile = File(...)):
    """
    Endpoint to remove an object from an image using a mask and the LaMa model.
    """
    logging.info(f"Received request for /inpaint for file: {file.filename}")

    if not lama_model:
        logging.error("Inpaint request failed because the LaMa model is not available.")
        raise HTTPException(status_code=503, detail="Inpainting model is not available.")

    try:
        image_bytes = await file.read()
        mask_bytes = await mask_file.read()

        if not image_bytes or not mask_bytes:
            logging.warning("Request failed because image or mask data is empty.")
            raise HTTPException(status_code=400, detail="Empty file or mask uploaded.")

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")

        logging.info("Performing inpainting with the LaMa model...")
        inpainted_image = lama_model(original_image, mask_image)
        logging.info("Inpainting completed successfully.")

        byte_io = io.BytesIO()
        inpainted_image.save(byte_io, 'PNG')
        byte_io.seek(0)

        return StreamingResponse(byte_io, media_type="image/png")

    except Exception as e:
        logging.error(f"An error occurred during inpainting: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
