Results





## Results Summary

### Audio Preprocessing & Feature Extraction

The audio pipeline was validated using a synthetic audio sample.

- Sample rate: **16,000 Hz**
- Audio length: ~62 seconds

#### Extracted Features

| Feature Type        | Shape (Features × Time) | Description |
|--------------------|--------------------------|-------------|
| Mel Spectrogram    | 128 × 1949               | Time–frequency representation used for CNN input |
| MFCC               | 40 × 1949                | Compact cepstral features for comparison |



<img width="1366" height="590" alt="image" src="https://github.com/user-attachments/assets/77d49380-8b9b-4a0d-86bd-f098f1bbe747" />


These features confirm successful audio preprocessing and are directly used as input to the CNN-based audio deepfake detection model.



### Visualization
Mel Spectrogram and MFCC visualizations were generated to verify spectral patterns and feature quality.




The model was evaluated using:

✅ Evaluation Results

Accuracy : 0.8
Precision: 0.8
Recall   : 0.8
F1-Score : 0.8


<img width="439" height="393" alt="image" src="https://github.com/user-attachments/assets/6cd37804-f2b7-45ba-80dc-bd3a197871ac" />



Sample performance:

\- Accuracy: ~80%

\- Balanced precision and recall



Classification Report:

              precision    recall  f1-score   support

        Real       0.80      0.80      0.80         5
        Fake       0.80      0.80      0.80         5

    accuracy                           0.80        10
   macro avg       0.80      0.80      0.80        10
weighted avg       0.80      0.80      0.80        10



Detailed evaluation was performed in Jupyter/Colab notebooks.



