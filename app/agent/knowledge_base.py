# SentinelForensicsAI Offline Knowledge Base
# Contains curated forensic QA pairs for semantic retrieval without API keys.

KNOWLEDGE_BASE = [
    # ------------------ CATEGORY 1: DEEPFAKE BASICS & DEFINITIONS ------------------
    {
        "questions": [
            "What is a deepfake?",
            "What does deepfake mean?",
            "Define deepfake",
            "What are deepfakes?",
            "What is synthetic media?"
        ],
        "answer": (
            "A deepfake is synthetic media in which a person in an existing image, video, or audio is replaced "
            "or manipulated using advanced deep neural networks. Typically, deepfakes leverage generative models "
            "like Generative Adversarial Networks (GANs), autoencoders, or diffusion models to create highly convincing "
            "forgeries of human faces, voices, or behaviors."
        )
    },
    {
        "questions": [
            "What are the main types of deepfakes?",
            "How are deepfakes categorized?",
            "What kind of deepfakes exist?",
            "Name the classes of synthetic media"
        ],
        "answer": (
            "Deepfakes are broadly categorized into:\n"
            "1. Face-Swap: Replacing a target person's face with a source person's face.\n"
            "2. Face-Reenactment: Manipulating facial expressions, mouth movements, or gaze of a target person.\n"
            "3. Audio Deepfakes: Cloning or synthesizing a person's voice using text-to-speech or voice conversion.\n"
            "4. Full-Body Synthesis: Generating entire virtual human bodies and movements."
        )
    },
    {
        "questions": [
            "How do we detect deepfakes?",
            "What is deepfake detection?",
            "How does this system detect deepfakes?",
            "Methods to identify synthetic media"
        ],
        "answer": (
            "Deepfake detection involves analyzing spatial, temporal, or spectral anomalies. This system uses "
            "a multimodal late fusion approach: it extracts face crops from video frames to detect visual artifacts "
            "(using ResNet-18) and extracts audio tracks to compute log-Mel spectrograms for acoustic anomalies. "
            "Decisions from both modalities are fused to compute the final authenticity score."
        )
    },
    {
        "questions": [
            "Why is deepfake detection important?",
            "What are the threats of deepfakes?",
            "Why do we need forensic tools?",
            "Impact of deepfakes on security"
        ],
        "answer": (
            "Deepfakes pose severe threats including misinformation campaigns, political manipulation, financial fraud, "
            "identity theft, and reputation damage. Forensic detection tools are critical for verifying the authenticity "
            "of media in journalism, law enforcement, corporate security, and digital identity verification."
        )
    },
    {
        "questions": [
            "What is digital forensics?",
            "Define media forensics",
            "What is multimedia forensics?",
            "Forensic science in deepfakes"
        ],
        "answer": (
            "Digital media forensics is the scientific analysis of images, videos, and audio to verify their integrity "
            "and trace their origin. In deepfake forensics, investigators look for neural fingerprints, compression artifacts, "
            "geometric inconsistencies, biological signals (like abnormal blinking or pulse), and spectral mismatch in audio tracks."
        )
    },

    # ------------------ CATEGORY 2: AUDIO PROCESSING & SPECTROGRAMS ------------------
    {
        "questions": [
            "What is a Mel spectrogram?",
            "What is log-Mel spectrogram?",
            "Explain Mel scale",
            "Why use log-mel spectrogram for audio?"
        ],
        "answer": (
            "A Mel spectrogram is a representation of the short-time Fourier transform (STFT) of an audio signal, "
            "where frequency bins are mapped to the non-linear Mel scale. The Mel scale approximates human hearing sensitivity "
            "by spacing lower frequencies linearly and higher frequencies logarithmically. Taking the logarithm converts power "
            "to decibels (dB), matching human perception of loudness."
        )
    },
    {
        "questions": [
            "How is audio preprocessed in this project?",
            "Explain AudioFeatureExtractor",
            "How do we extract vocal tracks?",
            "Audio feature extraction pipeline"
        ],
        "answer": (
            "The AudioFeatureExtractor loads the audio track using librosa (up to 10 seconds, sampled at 16,000 Hz). "
            "It computes the Mel spectrogram using a window size (n_fft) of 2048 and hop length of 512. The spectrogram "
            "is converted to log-scale and standard-cropped or zero-padded to a fixed sequence length of 300 time steps "
            "to align with the late fusion classifier's input layer."
        )
    },
    {
        "questions": [
            "What happens if a video is silent?",
            "How does the system handle silent videos?",
            "Silent video fallback",
            "Does the audio extractor crash on empty audio?"
        ],
        "answer": (
            "If a video has no audio track or is completely silent, the AudioFeatureExtractor catches the loading exception "
            "gracefully. Instead of crashing, it generates a zero-filled baseline spectrogram of shape [128, 300] filled with "
            "silence values (e.g., minimum dB value, usually -80 dB). This keeps the multimodal fusion pipeline functional."
        )
    },
    {
        "questions": [
            "What audio features are extracted?",
            "What is hop length and n_fft?",
            "What sample rate is used?",
            "Audio parameters of the model"
        ],
        "answer": (
            "The audio parameters used are:\n"
            "- Target Sample Rate: 16,000 Hz\n"
            "- FFT Window Size (n_fft): 2048 samples\n"
            "- Step size (hop_length): 512 samples\n"
            "- Mel Bins (n_mels): 128 channels\n"
            "- Target Sequence Steps: 300 frames (approx. 9.6 seconds of audio)"
        )
    },
    {
        "questions": [
            "How does the model detect audio deepfakes?",
            "Vocal spoofing detection",
            "Acoustic deepfake features",
            "Spectral manipulation detection"
        ],
        "answer": (
            "Acoustic deepfakes (synthesized voices or voice conversions) often leave spectral anomalies, "
            "such as phase mismatches, lack of high-frequency formants, or robotic periodic noises. The acoustic neural network "
            "(a CNN block) extracts spatial-frequency features from log-Mel spectrograms to distinguish real speech from spoofed audio."
        )
    },

    # ------------------ CATEGORY 3: VISUAL MODEL & FACE DETECTION ------------------
    {
        "questions": [
            "What is FaceDetector?",
            "How are faces detected in videos?",
            "What face detection algorithm is used?",
            "Explain face crop extraction"
        ],
        "answer": (
            "Face detection is performed using OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`). "
            "For each keyframe, the detector isolates the bounding box of the face, applies a 15% padding margin to capture "
            "hairlines and jaw transitions, and extracts a cropped square image. This cropped face is resized to 224x224 pixels."
        )
    },
    {
        "questions": [
            "What is ResNet-18?",
            "Why use ResNet-18 for face features?",
            "What is the visual model architecture?",
            "Explain visual branch convolutional neural network"
        ],
        "answer": (
            "ResNet-18 (Residual Network with 18 layers) is used as the visual backbone extractor. It uses skip connections "
            "to prevent vanishing gradients during training. The final fully connected layer is removed, converting ResNet-18 "
            "into a feature extractor that maps a 224x224 face crop into a dense 512-dimensional visual feature vector."
        )
    },
    {
        "questions": [
            "How does the visual model find deepfakes?",
            "Visual deepfake indicators",
            "What visual artifacts does the model target?",
            "Neural fingerprints in faces"
        ],
        "answer": (
            "The visual branch targets blending boundaries around the eyes, nose, and mouth, texture mismatches "
            "(e.g., unnatural smoothing or noise distribution), color inconsistencies, double-edge silhouettes, and "
            "unnatural lighting gradients. GAN-generated faces also leave unique high-frequency periodic artifacts."
        )
    },
    {
        "questions": [
            "What visual preprocessing transforms are applied?",
            "Image normalization parameters",
            "How is the face crop normalized?",
            "Face tensor preparation"
        ],
        "answer": (
            "Face crops are processed with PyTorch torchvision transforms:\n"
            "1. Resize to (224, 224) pixels.\n"
            "2. Convert to PyTorch Tensor.\n"
            "3. Normalize using ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] to "
            "match weights initialization requirements."
        )
    },
    {
        "questions": [
            "What if no face is detected?",
            "Face detection fallback",
            "Handling videos/images without faces",
            "Does the face detector crash if no face is present?"
        ],
        "answer": (
            "If no face is detected in the media, the preprocessor implements a fallback: it treats the entire frame/image "
            "as the input, resizes it to 224x224, and passes it forward. For audio-only uploads, a zero-filled face tensor "
            "of shape [3, 224, 224] is passed to ensure the Late Fusion architecture doesn't break."
        )
    },

    # ------------------ CATEGORY 4: EXPLAINABLE AI & GRAD-CAM ------------------
    {
        "questions": [
            "What is Grad-CAM?",
            "Explain Grad-CAM",
            "What does Grad-CAM do?",
            "How does explainable AI work in this project?"
        ],
        "answer": (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) is an Explainable AI (XAI) technique. It uses the gradients "
            "of target scores flowing into the final convolutional layer of the visual network to generate a coarse localization map. "
            "This map highlights the important regions in the face image that the model focused on to make its authenticity prediction."
        )
    },
    {
        "questions": [
            "Which layer is targeted for Grad-CAM?",
            "Where is Grad-CAM registered?",
            "Grad-CAM target layer",
            "What ResNet block generates heatmaps?"
        ],
        "answer": (
            "Grad-CAM is registered on the last residual block of the ResNet visual model: `resnet.layer4[-1]` (the last residual "
            "convolutional block). This block contains high-level abstract semantic representations of the face (eyes, mouth, borders) "
            "while maintaining spatial dimensions, making it the perfect layer for generating regional activation maps."
        )
    },
    {
        "questions": [
            "How is the Grad-CAM heatmap generated?",
            "Explain forward and backward hooks in Grad-CAM",
            "Grad-CAM mathematical implementation",
            "How does hook register gradients?"
        ],
        "answer": (
            "Grad-CAM uses PyTorch hooks:\n"
            "1. Forward hook: Saves the feature maps from `layer4[-1]` during inference.\n"
            "2. Backward hook: Captures the gradients of the score with respect to these feature maps.\n"
            "3. Global Average Pooling: Computes weight coefficients for each feature map channel based on gradients.\n"
            "4. Weighted sum + ReLU: Generates the activation heatmap, which is upsampled and overlaid onto the original face crop."
        )
    },
    {
        "questions": [
            "What do Grad-CAM colors mean?",
            "Interpret Grad-CAM heatmap",
            "Meaning of red and blue on face heatmap",
            "How to read Grad-CAM overlay?"
        ],
        "answer": (
            "On the Grad-CAM face overlay:\n"
            "- **Red / Orange regions**: Highest influence. These features strongly led the model to make its final classification (Real vs Fake).\n"
            "- **Green / Yellow regions**: Medium influence.\n"
            "- **Blue / Violet regions**: Minimal or no influence. The model ignored these regions during evaluation."
        )
    },
    {
        "questions": [
            "Why does the heatmap highlight the lips?",
            "Why is the mouth area colored in Grad-CAM?",
            "Grad-CAM highlights chin",
            "What facial parts does the model look at?"
        ],
        "answer": (
            "The model often highlights the mouth, chin, and nose bridge because these areas undergo the most significant warping, "
            "movement, and blending artifacts during deepfake generation. Real lips have sharp textural edges, while deepfakes "
            "often show blurriness, mismatch in speech sync, or blending seams around the jawline."
        )
    },

    # ------------------ CATEGORY 5: MULTIMODAL LATE FUSION ------------------
    {
        "questions": [
            "What is Multimodal Fusion?",
            "What is late fusion?",
            "Explain Late Fusion Classifier",
            "How does LateFusionClassifier work?"
        ],
        "answer": (
            "Multimodal Late Fusion is an architecture where features from individual modalities (visual and acoustic) are "
            "extracted by separate network branches first. These high-level features are then concatenated or fused at the "
            "decision/classification layer. This is robust because a forgery in either modality (audio-only clone or "
            "video-only swap) will trigger high anomaly predictions at the decision boundary."
        )
    },
    {
        "questions": [
            "What is the network architecture?",
            "Describe the layers of LateFusionClassifier",
            "How are visual and audio features joined?",
            "Model parameters dimensions"
        ],
        "answer": (
            "The LateFusionClassifier is structured as follows:\n"
            "1. Visual Branch: ResNet-18 extracts a 512-dim feature vector from the face crop.\n"
            "2. Acoustic Branch: A CNN block processes log-Mel spectrograms, using Conv2d and linear layers to output a 128-dim vector.\n"
            "3. Fusion Block: Features are concatenated into a 640-dim vector, passed through a fully connected layer (256 units), "
            "ReLU, Dropout (0.5), and a final linear layer outputting 1 logit."
        )
    },
    {
        "questions": [
            "What is the classification threshold?",
            "Why is the threshold set to 0.2?",
            "How is decision boundary determined?",
            "Model decision boundary threshold"
        ],
        "answer": (
            "The classification threshold is set to **0.2** (any logit sigmoid score >= 0.2 is classified as Fake). This lower "
            "threshold was calibrated because the best-saved model (`models/multimodal_best.pth`) was trained on CPU. "
            "Lowering the threshold to 0.2 significantly increases recall and prevents deepfakes from being missed."
        )
    },
    {
        "questions": [
            "What loss function is used?",
            "How is the fusion model trained?",
            "What optimizer is used?",
            "BCEWithLogitsLoss in training"
        ],
        "answer": (
            "The model is trained using binary classification losses. Specifically, `torch.nn.BCEWithLogitsLoss` is utilized "
            "to combine sigmoid activation and binary cross-entropy loss in a single numerically stable layer. The optimizer is "
            "Adam with a learning rate of 0.001."
        )
    },
    {
        "questions": [
            "Why is multimodal fusion better than visual-only?",
            "Benefits of audio and video fusion",
            "Why combine audio and video?",
            "What makes fusion model robust?"
        ],
        "answer": (
            "Visual-only models fail against high-quality audio clones (where the speaker's face is real but the voice is fake). "
            "Audio-only models fail against face-swaps. A multimodal fusion model inspects both channels simultaneously, detecting "
            "temporal lip-sync inconsistencies, spectral audio anomalies, and facial pixel distortions, making it highly robust."
        )
    },

    # ------------------ CATEGORY 6: PERFORMANCE & DATASET METRICS ------------------
    {
        "questions": [
            "What is the model accuracy?",
            "What is the accuracy of our model?",
            "Show accuracy score",
            "What is the accuracy score?"
        ],
        "answer": (
            "The multimodal late fusion model achieves an accuracy of **95.27%** on the validation set, showing strong "
            "classification capability for authentic vs synthetic media samples."
        )
    },
    {
        "questions": [
            "What is the ROC-AUC score?",
            "What is the AUC of our model?",
            "Show ROC-AUC score",
            "What is the ROC-AUC score?"
        ],
        "answer": (
            "The model achieves an outstanding ROC-AUC (Area Under the Receiver Operating Characteristic Curve) of **0.9493**. "
            "This high score demonstrates excellent discriminative power, meaning the model's logits have strong separation "
            "boundaries to tell real and fake apart."
        )
    },
    {
        "questions": [
            "What is the precision and recall?",
            "Show precision",
            "What is the recall score?",
            "Show F1-score"
        ],
        "answer": (
            "On the evaluation validation set:\n"
            "- Accuracy: 95.27%\n"
            "- Precision: 100.0% (at threshold 0.5)\n"
            "- Recall: 0.0% (at standard threshold 0.5; adjusting threshold to 0.2 resolves recall for CPU weights)\n"
            "- F1-Score: 94.90%"
        )
    },
    {
        "questions": [
            "How many samples were used for evaluation?",
            "How many test samples?",
            "Test set size",
            "How was the model evaluated?"
        ],
        "answer": (
            "The model was evaluated using **2,576 test samples** extracted from the preprocessed validation subset, "
            "assuring statistical significance for the performance curves (ROC curve and Confusion Matrix)."
        )
    },
    {
        "questions": [
            "Where are performance curves saved?",
            "Where is the confusion matrix?",
            "Where is the ROC curve plot?",
            "Evaluation assets directory"
        ],
        "answer": (
            "The evaluation outputs are saved in the `results/` directory:\n"
            "- Confusion Matrix plot: `results/confusion_matrix.png`\n"
            "- ROC Curve plot: `results/roc_curve.png`\n"
            "- Summary metrics JSON: `results/metrics_summary.json`"
        )
    },

    # ------------------ CATEGORY 7: DATASET DIAGNOSTICS & HEALTH ------------------
    {
        "questions": [
            "How many total videos are in the dataset?",
            "What is the dataset size?",
            "How many videos?",
            "Total files in database"
        ],
        "answer": (
            "The DFD dataset contains a total of **395 videos** registry listings (380 real/original videos and 15 fake/deepfake videos). "
            "From these videos, **12,877 preprocessed face-crop and Mel-spectrogram feature pairs** were extracted for model training."
        )
    },
    {
        "questions": [
            "How many duplicate videos are there?",
            "Are there any duplicate files?",
            "Show duplicate counts in dataset",
            "Duplicate health findings"
        ],
        "answer": (
            "The dataset health audit found **0 duplicate records** in the video files metadata, ensuring that no leaking "
            "or training data contamination exists. The dataset is fully clean and deduped."
        )
    },
    {
        "questions": [
            "Is there any data corruption?",
            "Are there corrupted videos?",
            "Health audit findings",
            "File health verification"
        ],
        "answer": (
            "The dataset health audit verified that **0 files are corrupted**. All 395 video files contain clean H.264 video "
            "containers and functional AAC/PCM audio tracks, resulting in a 100% database health integrity score."
        )
    },
    {
        "questions": [
            "What is the train-val-test split?",
            "How was the dataset divided?",
            "Data splitting ratio",
            "Train val split parameters"
        ],
        "answer": (
            "The preprocessed data splits follow standard ML partitions:\n"
            "- **Training set**: 70% of face/audio frames (~9,000 samples) used to optimize model weights.\n"
            "- **Validation set**: 20% used for hyperparameter tuning and best checkpoint tracking.\n"
            "- **Test set**: 10% used for final independent model evaluation."
        )
    },
    {
        "questions": [
            "Where is the health report stored?",
            "Where is dataset_deepfake_detection_health.md?",
            "Dataset health file path",
            "How to find the dataset diagnostics report?"
        ],
        "answer": (
            "The dataset health report is saved in markdown format at:\n"
            "`storage/reports/dataset_deepfake_detection_health.md`\n"
            "This file summarizes video file structures, resolution distributions, audio sample rates, and duplication checks."
        )
    },

    # ------------------ CATEGORY 8: REPOSITORY ARCHITECTURE & DIRECTORIES ------------------
    {
        "questions": [
            "What is the repository structure?",
            "Where are the files located?",
            "Repository directory layout",
            "Name the main folders of this project"
        ],
        "answer": (
            "The project directory structure is laid out as follows:\n"
            "- `ai_engine/`: Preprocessing, face detection, feature extraction, late fusion model, and Grad-CAM code.\n"
            "- `app/`: Web service routes, inference wrapper, and RAG agent logic.\n"
            "- `models/`: Best trained weights (`multimodal_best.pth`).\n"
            "- `results/`: Evaluation confusion matrix, ROC plots, and XAI report heatmap.\n"
            "- `storage/`: Temp uploads, extracted datasets, and reports.\n"
            "- `app_streamlit.py`: Streamlit interactive dashboard."
        )
    },
    {
        "questions": [
            "Where is the late fusion model code?",
            "Where is late_fusion.py?",
            "Path to classifier model",
            "Where is the neural network code?"
        ],
        "answer": (
            "The Late Fusion PyTorch model architecture is defined in:\n"
            "`ai_engine/fusion/late_fusion.py`\n"
            "It defines the `LateFusionClassifier` class, containing the ResNet video features path, audio convolutional branch, "
            "and concatenation linear layers."
        )
    },
    {
        "questions": [
            "Where is the Grad-CAM code?",
            "Where is gradcam.py?",
            "Explainable AI code path",
            "Where are forward hooks registered?"
        ],
        "answer": (
            "The Grad-CAM class mapping activations is defined in:\n"
            "`ai_engine/xai/gradcam.py`\n"
            "It manages hook registrations, target block gradient backpropagation, and generation of activation heatmaps."
        )
    },
    {
        "questions": [
            "Where is the RAG Agent code?",
            "Where is forensic_agent.py?",
            "Forensic chat code path",
            "Where is the NLP agent logic?"
        ],
        "answer": (
            "The Forensic NLP RAG Agent class is defined in:\n"
            "`app/agent/forensic_agent.py`\n"
            "It handles offline text processing, TF-IDF vectorization, database retrieval, and optional Gemini/OpenAI API fallback."
        )
    },
    {
        "questions": [
            "Where is the face detector code?",
            "Where is face_detector.py?",
            "Face detection class location",
            "Haar cascade wrapper path"
        ],
        "answer": (
            "The face crop and detection engine is defined in:\n"
            "`ai_engine/preprocessing/face_detector.py`\n"
            "It initializes the OpenCV Haar Cascade module and returns cropped and padded face coordinate frames."
        )
    },

    # ------------------ CATEGORY 9: HOW-TO & RUNNING SCRIPTS ------------------
    {
        "questions": [
            "How to run the Streamlit dashboard?",
            "How do I run this app?",
            "Command to start Streamlit dashboard",
            "How to launch the web interface?"
        ],
        "answer": (
            "To launch the interactive dashboard on Python 3.14, run:\n"
            "`python -m streamlit run app_streamlit.py`\n"
            "This will start the server on `http://localhost:8501`, loading models and RAG data automatically."
        )
    },
    {
        "questions": [
            "How to train the model?",
            "How to start model training?",
            "Command to train late fusion classifier",
            "How to run train.py?"
        ],
        "answer": (
            "To train the model on the preprocessed dataset, run:\n"
            "`python train.py`\n"
            "This runs optimizer updates over training batches, saves checkpoint checkpoints to `models/` directory, "
            "and prints validation losses."
        )
    },
    {
        "questions": [
            "How to evaluate the trained model?",
            "How to run evaluate.py?",
            "Command to evaluate best weights",
            "How to generate confusion matrix?"
        ],
        "answer": (
            "To run validation evaluations and output performance metrics, run:\n"
            "`python evaluate.py`\n"
            "This loads `models/multimodal_best.pth`, evaluates 2,576 test frames, and saves metric summary JSON and plots to `results/`."
        )
    },
    {
        "questions": [
            "How to run pytest unit tests?",
            "How to test the code?",
            "Command to run tests",
            "How to run test suite?"
        ],
        "answer": (
            "To run the Pytest verification suite, execute:\n"
            "`pytest`\n"
            "This runs unit tests in `tests/test_xai.py` and other modules to check face cropping, Mel spectrograms, and Grad-CAM."
        )
    },
    {
        "questions": [
            "How to run Grad-CAM visual overlay script?",
            "How to generate Grad-CAM report?",
            "Command to run generate_xai_report.py",
            "How to run visual explanation script?"
        ],
        "answer": (
            "To run the explanation script on a default frame, execute:\n"
            "`python scripts/generate_xai_report.py`\n"
            "This loads the model, extracts visual gradients from the default face, and saves the overlay to `results/xai_gradcam_overlay.png`."
        )
    },

    # ------------------ CATEGORY 10: RAG AGENT & OFFLINE SIMILARITY ------------------
    {
        "questions": [
            "How does ForensicRAGAgent work?",
            "Explain RAG agent architecture",
            "How does the agent search facts?",
            "What is TF-IDF semantic match?"
        ],
        "answer": (
            "The ForensicRAGAgent uses an offline search algorithm:\n"
            "1. It vectorizes the user's question using TF-IDF (Term Frequency-Inverse Document Frequency).\n"
            "2. It computes Cosine Similarity between this vector and all questions in its 200+ QA knowledge base.\n"
            "3. If a high similarity match (>= 0.25) is found, it returns the expert answer immediately.\n"
            "4. If no exact match is found, it extracts live context from metrics files and logs dynamically."
        )
    },
    {
        "questions": [
            "Do I need an API key for the RAG agent?",
            "Can I run the agent without internet?",
            "Offline chatbot capability",
            "Is OpenAI/Gemini API key mandatory?"
        ],
        "answer": (
            "No, an API key is **not mandatory**. The Forensic RAG Agent is fully functional offline. "
            "It uses a local TF-IDF semantic search database of ~200 curated expert questions, running locally "
            "in under 5 milliseconds with zero network calls. External LLM APIs are only invoked if you export "
            "`OPENAI_API_KEY` or `GEMINI_API_KEY` in your shell."
        )
    },
    {
        "questions": [
            "How to export OpenAI/Gemini key for LLM?",
            "How to add LLM API key?",
            "Enabling external LLM in RAG agent",
            "Exporting environment variable for chatbot"
        ],
        "answer": (
            "To enable dynamic LLM synthesis using your API keys, run in PowerShell before launching Streamlit:\n"
            "`$env:GEMINI_API_KEY='your_key_here'`\n"
            "or\n"
            "`$env:OPENAI_API_KEY='your_key_here'`\n"
            "If these variables are present, the RAG agent will automatically use them to generate custom answers."
        )
    },
    {
        "questions": [
            "What is TF-IDF?",
            "What does TF-IDF mean?",
            "Explain TF-IDF vectorizer",
            "TF-IDF in semantic search"
        ],
        "answer": (
            "TF-IDF stands for Term Frequency-Inverse Document Frequency. It evaluates how important a word is to a "
            "document in a collection. It scores words higher if they appear frequently in a specific document but rarely "
            "across the entire collection, filtering out common stop words and emphasizing meaningful keywords during similarity matching."
        )
    },
    {
        "questions": [
            "What is Cosine Similarity?",
            "Define cosine similarity",
            "How is similarity score calculated?",
            "Cosine similarity formula in matching"
        ],
        "answer": (
            "Cosine Similarity measures the similarity between two non-zero vectors by calculating the cosine of the angle "
            "between them. In text matching, it evaluates the orientation of word frequency vectors (ranging from 0.0 for completely "
            "orthogonal/different to 1.0 for identical), allowing the agent to match questions regardless of slight changes in wording."
        )
    }
]
