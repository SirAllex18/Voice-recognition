import os
import numpy as np
import librosa
import json
from sklearn.model_selection import train_test_split

def extract_mfcc(
    wav_path,
    sr=16000,
    n_mfcc=40,
    max_len=300
):
    """
    Load a .wav file and extract MFCC + delta + delta-delta features.
    """
    y, orig_sr = librosa.load(wav_path, sr=None)
    
    if sr and orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr
    
    mfcc = librosa.feature.mfcc(y=y, sr=orig_sr, n_mfcc=n_mfcc)
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    combined = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)

    combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-9)
    
    if combined.shape[1] < max_len:
        pad_width = max_len - combined.shape[1]
        combined = np.pad(
            combined, ((0, 0), (0, pad_width)),
            mode='constant'
        )
    else:
        combined = combined[:, :max_len]

    return combined

def load_data(data_root, sr=16000, n_mfcc=40, max_len=300):
    """
    Recursively walk through the VoxCeleb-style directory and extract MFCCs.
    data_root can be 'data/training_data' or 'data/testing_data'.
    
    You described the structure as:
        data_root/         (e.g. 'data/training_data')
          └─ wav/
             ├─ id10001/
             │   └─ 1zchlwhmdeo4/
             │       ├─ 00001.wav
             │       ├─ ...
             ├─ id10002/
             │   └─ ...
             └─ ...
    """
    X = []
    y = []
    
    label_dict = {}     
    label_counter = 0
    
    wav_folder = os.path.join(data_root, "wav")
    if not os.path.isdir(wav_folder):
        print(f"No 'wav' folder found inside {data_root}. Check directory structure.")
        return np.array([]), np.array([]), {}
    

    id_folders = os.listdir(wav_folder)
    
    for speaker_id_folder in id_folders:

        speaker_id_path = os.path.join(wav_folder, speaker_id_folder)
        
        if not os.path.isdir(speaker_id_path):
            continue
        
        if speaker_id_folder not in label_dict:
            label_dict[speaker_id_folder] = label_counter
            label_counter += 1
        
        speaker_label = label_dict[speaker_id_folder]
        
        subfolders = os.listdir(speaker_id_path)
        
        for subfolder_name in subfolders:
            subfolder_path = os.path.join(speaker_id_path, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            for wav_file in os.listdir(subfolder_path):
                if wav_file.endswith(".wav"):
                    wav_path = os.path.join(subfolder_path, wav_file)
                    
                
                    mfcc_features = extract_mfcc(
                        wav_path,
                        sr=sr,
                        n_mfcc=n_mfcc,
                        max_len=max_len
                    )
                    
                    X.append(mfcc_features)
                    y.append(speaker_label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} samples from {data_root}.")
    return X, y, label_dict

if __name__ == "__main__":
    training_data_dir = "data/training_data"
    testing_data_dir  = "data/testing_data"
    
    print("Loading training data...")
    X_train, y_train, train_label_dict = load_data(training_data_dir)
    
 
    val_ratio = 0.15  

    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=42
    )

    print("Train part shape:", X_train_part.shape)
    print("Val shape:", X_val.shape)
    
    print("Loading testing data...")
    X_test, y_test, test_label_dict = load_data(testing_data_dir)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    
   
    import json
    with open("label_map.json", "w") as f:
        json.dump(train_label_dict, f)

    np.save("X_train_part.npy", X_train_part)
    np.save("y_train_part.npy", y_train_part)

    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)

    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print("Data saved! Train/Val/Test shapes:")
    print("  train:", X_train_part.shape, y_train_part.shape)
    print("  val:  ", X_val.shape, y_val.shape)
    print("  test: ", X_test.shape, y_test.shape)
