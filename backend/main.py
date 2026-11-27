from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai_engine import VascularAIEngine
from seed_samples import generate_reference_outputs

from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from PIL import Image

from io import BytesIO
import os
import base64
import numpy as np

if not os.path.exists("attention_mask.png"):
    generate_reference_outputs()

app = FastAPI(
    title="VascularAI Clinical Workstation API",
    description="Artery Segmentation, Centerline Extraction, Stenosis Severity & Grad-CAM++ API",
    version="3.0.0"
)

app.mount("/test-files", StaticFiles(directory="."), name="test-files")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_engine = VascularAIEngine()


def array_to_base64(arr: np.ndarray, mode: str = "RGB") -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode=mode)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


@app.get("/")
def home():
    return {"message": "VascularAI FastAPI backend is running", "status": "Connected"}


@app.get("/dashboard/summary")
def dashboard_summary():
    return {
        "backend_status": "Connected",
        "selected_model": "MultiScale Residual / Attention U-Net",
        "test_dice": 47.09,
        "validation_dice": 51.65,
    }


@app.post("/analyze-scan")
async def analyze_scan(
    file: UploadFile = File(...),
    model_type: str = Form("attention")
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        result = ai_engine.run_full_analysis(image_bytes, model_type=model_type)
        return {"status": "completed", "filename": file.filename, **result}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    return await analyze_scan(file=file, model_type="attention")


@app.post("/extract-centerline")
async def extract_centerline_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("L")
    mask_array = np.array(image)
    binary_mask = mask_array > 0
    centerline = skeletonize(binary_mask)
    cl_img = (centerline * 255).astype(np.uint8)
    return {
        "message": "Centerline extracted successfully",
        "width": int(centerline.shape[1]),
        "height": int(centerline.shape[0]),
        "centerline_pixels": int(centerline.sum()),
        "centerline_image": array_to_base64(cl_img, "L")
    }


@app.post("/analyze-vessel-width")
async def analyze_vessel_width(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("L")
    mask_array = np.array(image)
    binary_mask = mask_array > 0

    if not binary_mask.any():
        raise HTTPException(status_code=400, detail="The uploaded mask is empty.")

    centerline = skeletonize(binary_mask)
    distance_map = distance_transform_edt(binary_mask)
    y_coords, x_coords = np.where(centerline)

    if len(x_coords) == 0:
        raise HTTPException(status_code=400, detail="No centerline could be extracted.")

    widths = distance_map[y_coords, x_coords] * 2
    width_map = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    min_w, max_w = float(widths.min()), float(widths.max())
    w_range = max_w - min_w

    for x, y, w in zip(x_coords, y_coords, widths):
        norm = (w - min_w) / w_range if w_range > 0 else 0.5
        width_map[y, x] = [int(255 * (1 - norm)), int(255 * norm), 40]

    return {
        "message": "Vessel width analysis completed.",
        "width": int(mask_array.shape[1]),
        "height": int(mask_array.shape[0]),
        "centerline_pixels": int(len(widths)),
        "minimum_width_px": round(min_w, 2),
        "median_width_px": round(float(np.median(widths)), 2),
        "maximum_width_px": round(max_w, 2),
        "width_heatmap": array_to_base64(width_map, "RGB"),
        "measurement_unit": "pixels",
        "clinical_status": "Experimental geometric measurement; not a stenosis diagnosis."
    }
