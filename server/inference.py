import torch
import torch.nn as nn
import numpy as np
import librosa
import argparse
import soundfile as sf
import io
import os
from speaker_cnn import SpeakerCNN

def load_model(model_path: str, num_speakers: int, device: torch.device):
    """
    Create the SpeakerCNN, load its state_dict, and set to eval mode.
    """
    model = SpeakerCNN(num_speakers=num_speakers)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def extract_mfcc_inference(
    audio_input,
    sr=16000,
    n_mfcc=40,
    max_len=300,
    is_file_path=True
):
    if is_file_path:
        y, orig_sr = librosa.load(audio_input, sr=None)
    else:
        y, orig_sr = sf.read(io.BytesIO(audio_input))
    
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr
    
    mfcc = librosa.feature.mfcc(y=y, sr=orig_sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
    combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-9)
    
    if combined.shape[1] < max_len:
        pad_width = max_len - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :max_len]
    
    return combined

def predict_speaker(
    model: nn.Module,
    audio_input,
    device: torch.device,
    threshold: float = 0.7,
    reverse_label_map= None,
    num_mfcc: int = 40,
    max_len: int = 300,
    is_file_path=True
):
    combined = extract_mfcc_inference(
        audio_input=audio_input,
        sr=16000,
        n_mfcc=num_mfcc,
        max_len=max_len,
        is_file_path=is_file_path
    ) 

    input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)   
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    predicted_label = np.argmax(probs)
    confidence = probs[predicted_label]

    predicted_label = int(predicted_label)
    confidence = float(confidence)

    if confidence < threshold:
        return "unknown", confidence
    else:
        if reverse_label_map is not None:
            if predicted_label in reverse_label_map:
                label_str = reverse_label_map[predicted_label]
                return label_str, confidence
            else:
                return f"id{predicted_label}", confidence
        else:
            return predicted_label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to a .wav file.")
    parser.add_argument("--model_path", type=str, default="speaker_cnn_model.pth", help="Path to the trained model state_dict.")
    parser.add_argument("--num_speakers", type=int, default=202, help="Number of speaker classes in the model.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for 'unknown'.")
    parser.add_argument("--label_map_file", type=str, default="", help="Path to a label_map file (optional).")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model = load_model(args.model_path, num_speakers=args.num_speakers, device=device)
    
    label_map = None
    if args.label_map_file and os.path.exists(args.label_map_file):
        import json
        with open(args.label_map_file, 'r') as f:
            label_map_json = json.load(f)
        label_map = {int(k): v for k, v in label_map_json.items()}
    
    speaker_id_or_unknown, conf = predict_speaker(
        model=model,
        audio_input=args.audio_path,
        device=device,
        threshold=args.threshold,
        label_map=label_map,
        num_mfcc=40,
        max_len= 300,
        is_file_path=True
    )

    print(f"Prediction: {speaker_id_or_unknown}, confidence={conf:.3f}")
