import torch
import torch.nn as nn
import numpy as np
import librosa
import argparse
import soundfile as sf
import io
import os

class SpeakerCNN(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = None  
        self.fc2   = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x_flat = x.view(x.size(0), -1)
        
        if self.fc1 is None:
            in_features = x_flat.size(1)
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, num_speakers).to(x.device)

        x = torch.relu(self.fc1(x_flat))
        x = self.fc2(x)
        return x
    
def load_model(model_path: str, num_speakers: int, device: torch.device):
    """
    Create the SpeakerCNN, load its state_dict, and set to eval mode.
    """
    model = SpeakerCNN(num_speakers=num_speakers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_mfcc_inference(
    audio_input,
    sr=16000,
    n_mfcc=40,
    max_len=200,
    is_file_path=True
):
    """
    Extract MFCC features using the same parameters as loadData.py.

    Parameters
    ----------
    audio_input: str or bytes
        - If is_file_path=True, this is a path to a .wav file.
        - If is_file_path=False, this is a bytes object (raw audio).
    sr: int
        Target sampling rate (16k).
    n_mfcc: int
        Number of MFCC coefficients.
    max_len: int
        Fixed length for MFCC time steps.
    is_file_path: bool
        True if audio_input is a file path, False if audio_input is raw bytes.

    Returns
    -------
    mfcc : np.ndarray of shape (n_mfcc, max_len)
    """
    # 1. Load audio
    if is_file_path:
        y, orig_sr = librosa.load(audio_input, sr=None)
    else:
        y, orig_sr = sf.read(io.BytesIO(audio_input))
    
    # 2. Resample if needed
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr
    
    # 3. Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=orig_sr, n_mfcc=n_mfcc)
    
    # 4. Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    
    # 5. Pad or truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

def predict_speaker(
    model: nn.Module,
    audio_input,
    device: torch.device,
    threshold: float = 0.7,
    label_map: dict = None,
    num_mfcc: int = 40,
    max_len: int = 200,
    is_file_path=True
):
    """
    Predict the speaker ID or 'unknown' from a given audio input.
    
    Parameters
    ----------
    model : nn.Module
        Loaded SpeakerCNN model in eval mode.
    audio_input : str or bytes
        - If is_file_path=True, then a path to a .wav file.
        - Else, raw audio bytes (for an API).
    device : torch.device
        "cpu" or "cuda".
    threshold : float
        Probability threshold to decide if speaker is 'unknown'.
    label_map : dict
        Maps numeric labels -> speaker IDs, e.g. {0: "id10291", 1: "id10292", ...}.
        If None, will return the numeric label instead of a speaker ID string.
    num_mfcc : int
        Must match training extraction.
    max_len : int
        Must match training extraction.
    is_file_path : bool
        True if audio_input is a filepath, False if audio bytes.

    Returns
    -------
    predicted_label_or_unknown, confidence
    """

    # 1. Extract MFCC
    mfcc = extract_mfcc_inference(
        audio_input=audio_input,
        sr=16000,
        n_mfcc=num_mfcc,
        max_len=max_len,
        is_file_path=is_file_path
    ) 

    # 2. Convert to torch tensor => (1, 1, n_mfcc, max_len)
    input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)   
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # 4. Determine predicted label
    predicted_label = np.argmax(probs)
    confidence = probs[predicted_label]

    # 5. Check threshold for 'unknown'
    if confidence < threshold:
        return "unknown", float(confidence)
    else:
        if label_map is not None and predicted_label in label_map:
            return label_map[predicted_label], float(confidence)
        else:
            return predicted_label, float(confidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to a .wav file.")
    parser.add_argument("--model_path", type=str, default="speaker_cnn_model.pth", help="Path to the trained model state_dict.")
    parser.add_argument("--num_speakers", type=int, default=10, help="Number of speaker classes in the model.")
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
        max_len=200,
        is_file_path=True
    )

    print(f"Prediction: {speaker_id_or_unknown}, confidence={conf:.3f}")
