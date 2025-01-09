import os
import numpy as np
import librosa

def extract_mfcc(wav_path, sr=16000, n_mfcc=40, max_len=200):
    """
    Load a .wav file and extract MFCC features.
    
    Parameters:
    -----------
    wav_path : str
        Path to the audio file (.wav)
    sr : int
        Target sampling rate to resample the audio
    n_mfcc : int
        Number of MFCC coefficients to extract
    max_len : int
        Fixed length of MFCC time steps (for padding or truncation)
    
    Returns:
    --------
    mfcc : np.ndarray
        A 2D array of shape (n_mfcc, max_len) containing the MFCC features
    """
    # Load audio
    y, orig_sr = librosa.load(wav_path, sr=None) 
    if sr and orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        orig_sr = sr
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=orig_sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc


def load_data(data_root, sr=16000, n_mfcc=40, max_len=200):
    """
    Recursively walk through the dataset directory, 
    extract MFCCs for each .wav file, and build X, y arrays.

    Parameters:
    -----------
    data_root : str
        Path to either training_data or testing_data folder
    sr : int
        Sampling rate for audio
    n_mfcc : int
        Number of MFCC coefficients
    max_len : int
        Fixed length for MFCC frames

    Returns:
    --------
    X : np.ndarray
        Shape (num_samples, n_mfcc, max_len)
    y : np.ndarray
        Shape (num_samples,)
    label_dict : dict
        Dictionary mapping speaker_id string to numeric label
    """
    X = []
    y = []
    label_dict = {}
    label_counter = 0
    
  
    for speaker_id_folder in os.listdir(data_root):
       
        speaker_id_path = os.path.join(data_root, speaker_id_folder)
        
     
        if not os.path.isdir(speaker_id_path):
            continue
        
    
        if speaker_id_folder not in label_dict:
            label_dict[speaker_id_folder] = label_counter
            label_counter += 1
        
        speaker_label = label_dict[speaker_id_folder]
        
        
        for subfolder in os.listdir(speaker_id_path):
            subfolder_path = os.path.join(speaker_id_path, subfolder)
            
            if not os.path.isdir(subfolder_path):
                continue
            
           
            for wav_file in os.listdir(subfolder_path):
                if wav_file.endswith(".wav"):
                    wav_path = os.path.join(subfolder_path, wav_file)
                   
                    mfcc_features = extract_mfcc(wav_path, sr=sr, n_mfcc=n_mfcc, max_len=max_len)
                    
                   
                    X.append(mfcc_features)
                    y.append(speaker_label)
    
   
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} samples from {data_root}.")
    return X, y, label_dict


if __name__ == "__main__":

    training_data_dir = "data/training_data"
    testing_data_dir = "data/testing_data"
    
    print("Loading training data...")
    X_train, y_train, train_label_dict = load_data(training_data_dir)

    print("Loading testing data...")
    X_test, y_test, test_label_dict = load_data(testing_data_dir)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
