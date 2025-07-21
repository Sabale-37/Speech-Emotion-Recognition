
# 📢 Speech Emotion Recognition using Wav2Vec2 and Hugging Face Transformers

> A deep learning pipeline for **classifying emotions from speech audio** using the **Wav2Vec2** model with PyTorch and Hugging Face Transformers.



## 🎯 Project Objective
The objective of this project is to **classify human emotions** from audio signals using advanced **pre-trained speech representations**. It leverages **Wav2Vec2** to extract deep features from raw waveform data and fine-tunes it for classification tasks using labeled datasets of emotional speech.


## 📊 Dataset

### Toronto Emotional Speech Set (TESS)
- **Emotions Covered**: Happy, Sad, Fear, Disgust, Neutral, Angry, PS (custom label)
- **Number of Samples**: ~2800 audio files
- **Format**: `.wav`
- **Labels**: Extracted from file naming convention.

> Example:
```
OAF_happy.wav → Label: happy
YAF_sad.wav   → Label: sad
```

## 🛠️ Tech Stack

| Technology  | Purpose                        |
|-------------|--------------------------------|
| **Python**  | Core Programming Language      |
| **PyTorch** | Deep Learning Framework        |
| **Hugging Face Transformers** | Wav2Vec2 for Speech Feature Extraction & Classification |
| **Librosa** | Audio Processing                |
| **Matplotlib / Seaborn** | Data Visualization |
| **Scikit-learn** | Metrics & Splitting Data   |


## 📝 Key Steps / Workflow

### 1️⃣ **Data Loading & Labeling**
- Walk through directory structure.
- Extract labels from filenames.
- Store paths and labels in a Pandas DataFrame.

### 2️⃣ **EDA (Exploratory Data Analysis)**
- Count plot for class distribution.
- Visualization of waveform and spectrograms.

### 3️⃣ **Dataset Preparation**
- Labels are mapped to integers.
- Custom `Dataset` class for PyTorch defined.
- Audio processed using `Librosa` and Hugging Face `Wav2Vec2Processor`.

### 4️⃣ **Model Setup**
- Pre-trained **Wav2Vec2 (facebook/wav2vec2-base)** loaded.
- Final classification head adjusted for emotion classes.

### 5️⃣ **Training**
- Hugging Face `Trainer` API used.
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score.
- Training arguments configured: epochs, batch size, learning rate.

### 6️⃣ **Evaluation**
- Evaluate on test set.
- Compute weighted metrics for imbalanced classes.

### 7️⃣ **Prediction**
- Random test audio sample predicted.
- Outputs both original and predicted emotion.


## 🔮 Sample Prediction
```plaintext
Original Label: happy
Predicted Label: happy
```


## 📌 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/sabale-37/Speech-Emotion-Recognition.git
cd speech-emotion-recognition
```

### 2️⃣ Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook
```bash
jupyter notebook
```


## 📥 Dependencies (requirements.txt)
```
torch
transformers
datasets
scikit-learn
librosa
matplotlib
seaborn
pandas
numpy
ipython
```

## 📈 Future Improvements
- Hyperparameter tuning via Optuna or Grid Search.
- Augment dataset with noise-robust training.
- Compare transformer-based approaches with CNN-LSTM baselines.
- Real-time inference via Streamlit or Gradio interface.

## 🤝 Contributing
Feel free to submit issues or PRs. Contributions are welcome!

## 📜 License
[MIT License](LICENSE)

## 🧑‍💻 Author
**Narayan Sabale**  
**narayansabale026@gmail.com**
