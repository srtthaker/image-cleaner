import io
import logging
import traceback
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2

# --- Model Imports ---
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Photo Editor API (YOLOv8)",
    description="A robust API for background removal (YOLOv8) and inpainting (LaMa).",
    version="4.3.0",
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
try:
    logging.info("Loading YOLOv8 Extra-Large segmentation model...")
    yolo_model = YOLO('yolov8x-seg.pt')
    logging.info("YOLOv8 Extra-Large model loaded successfully.")
except Exception as e:
    logging.error(f"Fatal error loading YOLOv8 model: {e}")
    logging.error(traceback.format_exc())
    yolo_model = None

# Load the LaMa inpainting model
lama_model = None
try:
    logging.info("Attempting to load LaMa Inpainting model...")
    lama_model = SimpleLama()
    logging.info("LaMa Inpainting model loaded successfully.")
except Exception as e:
    logging.error(f"Fatal error loading LaMa model: {e}")
    logging.error(traceback.format_exc())


# --- Mask Refinement Logic ---
def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Refines a binary mask by filling small holes and smoothing the edges.
    This version is more aggressive to close larger gaps.
    """
    # Use a larger kernel for more aggressive morphological operations.
    kernel = np.ones((11, 11), np.uint8)
    
    # Perform a morphological closing operation with more iterations to fill larger holes.
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    # Apply a slightly larger Gaussian blur to feather the edges for a smooth transition.
    feathered_mask = cv2.GaussianBlur(closed_mask, (21, 21), 0)
    
    return feathered_mask


# --- Background Removal Logic ---
def remove_background_yolo(image_bytes: bytes):
    """
    Processes an image to remove the background using YOLOv8 segmentation.
    It specifically looks for people, combines their masks, and refines the result.
    """
    if not yolo_model:
        raise HTTPException(status_code=503, detail="YOLOv8 model is not available.")

    try:
        # Convert image bytes to a NumPy array that OpenCV can use
        original_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(original_image_pil)
        
        # Run the YOLOv8 model on the image
        # We are only interested in the 'person' class, which has an ID of 0
        results = yolo_model.predict(image_np, classes=[0], verbose=False)
        
        # Check if any masks were detected
        if not results[0].masks:
            logging.warning("No person detected in the image. Returning transparent image.")
            # If no person is found, create a fully transparent image of the same size
            h, w, _ = image_np.shape
            return Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # Create a combined mask for all detected people
        combined_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
        for mask_data in results[0].masks.data:
            # Convert mask tensor to numpy array and resize to original image dimensions
            mask_np = mask_data.cpu().numpy().astype(np.uint8)
            resized_mask = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))
            # Add the mask of the current person to the combined mask
            combined_mask = np.maximum(combined_mask, resized_mask * 255)

        # ** NEW STEP: Refine the combined mask to fill holes and smooth edges **
        refined_mask = refine_mask(combined_mask)

        # Create a 4-channel RGBA image from the original RGB image
        image_rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        
        # Apply the REFINED mask to the alpha channel
        image_rgba[:, :, 3] = refined_mask

        # Convert the final NumPy array back to a PIL Image
        output_image = Image.fromarray(image_rgba)
        
        return output_image

    except Exception as e:
        logging.error(f"An error occurred during YOLOv8 processing: {e}")
        logging.error(traceback.format_exc())
        raise e


# --- API Endpoints ---
@app.get("/")
def read_root():
    """ Root endpoint to check if the API is running. """
    return {"message": "Welcome to the AI Photo Editor API (YOLOv8)!"}


@app.post("/remove-bg")
async def remove_background_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to remove the background from an uploaded image using YOLOv8.
    """
    logging.info(f"Received request for /remove-bg for file: {file.filename}")
    try:
        input_image_bytes = await file.read()
        output_image_pil = remove_background_yolo(input_image_bytes)
        
        # Save the PIL image to a byte stream to send as a response
        byte_io = io.BytesIO()
        output_image_pil.save(byte_io, 'PNG')
        byte_io.seek(0)
        
        return StreamingResponse(byte_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/inpaint")
async def inpaint_image(file: UploadFile = File(...), mask_file: UploadFile = File(...)):
    """
    Endpoint to remove an object from an image using a mask and the LaMa model.
    """
    logging.info(f"Received request for /inpaint for file: {file.filename}")

    if not lama_model:
        raise HTTPException(status_code=503, detail="Inpainting model is not available.")

    try:
        image_bytes = await file.read()
        mask_bytes = await mask_file.read()

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")

        inpainted_image = lama_model(original_image, mask_image)

        byte_io = io.BytesIO()
        inpainted_image.save(byte_io, 'PNG')
        byte_io.seek(0)

        return StreamingResponse(byte_io, media_type="image/png")

    except Exception as e:
        logging.error(f"An error occurred during inpainting: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
