import uvicorn
from typing import Union
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch

from inference import load_model, predict_speaker

app = FastAPI()

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 1. Load your model & define label_map
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Suppose you know how many unique speakers (num_speakers). 
# For example, if you have 20 total speaker classes in your training.
NUM_SPEAKERS = 20

model_path = "speaker_cnn_model.pth"  # The file you saved after training
model = load_model(model_path, num_speakers=NUM_SPEAKERS, device=device)

# If you have a label map, define it here or load from JSON
label_map = {
    0: "id10291",
    1: "id10292",
    # ...
    # 19: "idXXXX"
}

@app.get("/")
def root():
    return {"message": "Server is running"}

@app.post("/recognize")
async def recognize_speaker(file: UploadFile = File(...)):
    """
    Endpoint to recognize speaker from a .wav file.
    Returns JSON with either {"speaker_id": "...", "confidence": ...}
    or {"status": "unknown_speaker", "confidence": ...}.
    """
    # 1. Read the uploaded file as bytes
    audio_bytes = await file.read()

    # 2. Run prediction
    speaker_label, confidence = predict_speaker(
        model=model,
        audio_bytes=audio_bytes,
        device=device,
        threshold=0.7,      # you can tweak
       # label_map=label_map # if you want textual labels
    )

    # 3. Return response
    if speaker_label == "unknown":
        return {"status": "unknown_speaker", "confidence": confidence}
    else:
        return {"speaker_id": speaker_label, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
