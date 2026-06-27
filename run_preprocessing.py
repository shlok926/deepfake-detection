import os
import numpy as np
import pandas as pd
from preprocess import extract_frames_and_audio, detect_and_crop_faces, extract_audio_features

def run_pipeline(raw_videos_dir="data/raw_videos", metadata_output_path="data/metadata.csv"):
    """
    Orchestrates the entire preprocessing pipeline for all videos in raw_videos_dir.
    Generates metadata.csv mapping face crops to audio features.
    """
    categories = ["real", "fake"]
    records = []
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    
    for category in categories:
        cat_dir = os.path.join(raw_videos_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Directory {cat_dir} does not exist. Skipping.")
            continue
            
        label = 0 if category == "real" else 1
        videos = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not videos:
            print(f"No videos found in {cat_dir}")
            continue
            
        print(f"Processing {len(videos)} videos in category: {category}...")
        
        for video_file in videos:
            video_path = os.path.join(cat_dir, video_file)
            base_name = os.path.splitext(video_file)[0]
            
            # Setup intermediate paths
            frame_dir = f"extracted_frames/{category}/{base_name}"
            audio_path = f"audio_data/{category}/{base_name}.wav"
            crop_dir = f"face_crops/{category}/{base_name}"
            mel_npy_path = f"audio_data/{category}/{base_name}_mel.npy"
            
            print(f"\n--- Processing: {video_file} ---")
            
            # 1. Extract Frames & Audio
            extract_frames_and_audio(video_path, frame_dir, audio_path)
            
            # 2. Crop Faces from Extracted Frames
            detect_and_crop_faces(frame_dir, crop_dir)
            
            # 3. Extract Audio Mel Spectrogram
            mel_spec = extract_audio_features(audio_path)
            
            if mel_spec is None:
                # If no audio, generate a dummy Mel Spectrogram (128 mels, 100 timesteps of zeros)
                print(f"Warning: No audio features found for {video_file}. Creating fallback dummy spectrogram.")
                mel_spec = np.zeros((128, 100))
            
            # Save Mel Spectrogram as .npy
            os.makedirs(os.path.dirname(mel_npy_path), exist_ok=True)
            np.save(mel_npy_path, mel_spec)
            print(f"Mel Spectrogram saved to {mel_npy_path}")
            
            # 4. Record mapping for each face crop generated
            if os.path.exists(crop_dir):
                crops = [f for f in os.listdir(crop_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if crops:
                    for crop in crops:
                        crop_path = os.path.join(crop_dir, crop)
                        records.append({
                            "video_name": video_file,
                            "face_path": crop_path,
                            "audio_mel_path": mel_npy_path,
                            "label": label
                        })
                    print(f"Recorded {len(crops)} face crops for {video_file}")
                else:
                    print(f"No faces detected for {video_file}")
            else:
                print(f"Crop directory {crop_dir} was not created.")
                
    # Save to CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv(metadata_output_path, index=False)
        print(f"\nPreprocessing finished! Metadata CSV saved to {metadata_output_path} with {len(df)} samples.")
    else:
        print("\nPreprocessing finished, but no samples were successfully processed.")

if __name__ == "__main__":
    # Create the folder structure if not already present
    os.makedirs("data/raw_videos/real", exist_ok=True)
    os.makedirs("data/raw_videos/fake", exist_ok=True)
    
    print("Preprocessing runner initialized.")
    # To run: python run_preprocessing.py
    run_pipeline()
