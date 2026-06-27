import os
import numpy as np
import pandas as pd
from PIL import Image

def generate_mock_dataset(output_dir="data", num_videos=10):
    """
    Generates dummy face crops, audio Mel spectrograms, and metadata.csv
    so you can test the training and evaluation pipeline immediately.
    """
    print("Generating mock dataset files for testing...")
    
    face_dir = os.path.join(output_dir, "..", "face_crops")
    audio_dir = os.path.join(output_dir, "..", "audio_data")
    
    categories = ["real", "fake"]
    records = []
    
    for category in categories:
        label = 0 if category == "real" else 1
        
        for v_idx in range(num_videos // 2):
            video_name = f"mock_{category}_video_{v_idx}"
            
            # 1. Create a dummy Mel spectrogram (.npy file)
            # Standard shape: (128, time_steps) where time_steps is variable (e.g., between 80 and 200)
            time_steps = np.random.randint(80, 200)
            dummy_mel = np.random.randn(128, time_steps)
            
            cat_audio_dir = os.path.join(audio_dir, category)
            os.makedirs(cat_audio_dir, exist_ok=True)
            
            mel_path = os.path.join(cat_audio_dir, f"{video_name}_mel.npy")
            np.save(mel_path, dummy_mel)
            
            # 2. Create 2-3 dummy face crop images (.jpg files)
            cat_face_dir = os.path.join(face_dir, category, video_name)
            os.makedirs(cat_face_dir, exist_ok=True)
            
            num_crops = np.random.randint(2, 4)
            for c_idx in range(num_crops):
                # Generate a random color image (224x224)
                random_image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(random_image_array)
                
                crop_path = os.path.join(cat_face_dir, f"frame_{c_idx:04d}_face0.jpg")
                img.save(crop_path)
                
                # Append to metadata record
                records.append({
                    "video_name": f"{video_name}.mp4",
                    "face_path": os.path.normpath(crop_path),
                    "audio_mel_path": os.path.normpath(mel_path),
                    "label": label
                })
                
    # Save metadata CSV
    metadata_path = os.path.join(output_dir, "metadata.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    
    print(f"Mock dataset generated successfully!")
    print(f"Saved {len(df)} sample links in {metadata_path}")
    print("Directories populated: 'face_crops/' and 'audio_data/'")

if __name__ == "__main__":
    generate_mock_dataset()
