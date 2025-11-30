import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from texteller import load_model, load_tokenizer, img2latex
import uvicorn
import os

import torch

app = FastAPI()

# Load model and tokenizer globally on startup
print("Loading TexTeller model...")
use_onnx = True
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"Using device: {device_str}, ONNX: {use_onnx}")

model = load_model(use_onnx=use_onnx)
tokenizer = load_tokenizer()
print("Model loaded.")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # img2latex expects a list of images
        # out_format="katex" might be better for web rendering if available, 
        # but the user asked for LaTeX. The doc says out_format="latex" or "katex".
        # Let's stick to "latex" for now, or "katex" if it helps rendering.
        # KaTeX is a rendering library, so getting "katex" compatible string is probably best.
        latex_list = img2latex(model, tokenizer, [image_np], out_format="katex", device=device)
        latex = latex_list[0]
        
        return {"latex": latex}
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
