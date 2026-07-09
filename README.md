# 🖼️ Image Captioning AI

An AI-powered Image Captioning application that automatically generates meaningful captions for images by combining **Computer Vision** and **Natural Language Processing**.

The project uses **VGG16** as a pre-trained Convolutional Neural Network (CNN) for image feature extraction and an **LSTM-based Recurrent Neural Network (RNN)** to generate natural language captions. A user-friendly **Streamlit** interface allows users to upload images and receive AI-generated captions in real time.

---

## 🚀 Features

- Upload any image through a Streamlit web application
- Automatically extracts image features using **VGG16**
- Generates natural language captions using an **LSTM Decoder**
- Trained on the **Flickr8k** dataset
- Clean and modular project structure
- Easy to train and customize
- Interactive and responsive web interface

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Deep Learning
- TensorFlow / Keras

### Computer Vision
- VGG16 (Pre-trained CNN)
- OpenCV
- Pillow

### Natural Language Processing
- Tokenizer
- Word Embeddings
- LSTM (Long Short-Term Memory)

### Web Framework
- Streamlit

### Data Processing
- NumPy
- Pandas

---

## 📂 Project Structure

```
Image_Captioning/
│
├── app.py
├── predict.py
├── train.py
├── build_tokenizer.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│
├── features/
│
├── models/
│
├── screenshots/
│
└── utils/
    ├── data_generator.py
    ├── feature_extractor.py
    ├── model_builder.py
    ├── preprocess.py
    └── tokenizer_utils.py
```

---

## 🧠 Model Architecture

```
Input Image
      │
      ▼
Pre-trained VGG16
      │
Image Features (4096)
      │
      ▼
LSTM Decoder
      │
      ▼
Generated Caption
```

---

## 📊 Dataset

**Dataset Used:** Flickr8k

- Approximately **8,000 images**
- Five captions per image
- More than **40,000 captions**
- Suitable for image captioning research

Download from Kaggle:

https://www.kaggle.com/datasets/nunenuh/flickr8k

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Ashishkumar217/Image_Captioning_AI.git
cd Image_Captioning_AI
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application

```bash
streamlit run app.py
```

---

## 🏋️ Training the Model

### Build Tokenizer

```bash
python build_tokenizer.py
```

### Extract Image Features

```bash
python utils/feature_extractor.py
```

### Train Model

```bash
python train.py
```

---

## 🖼️ Screenshots

### Home Page

> Add screenshot here

```
screenshots/home.png
```

### Caption Generation

> Add screenshot here

```
screenshots/prediction.png
```

---

## 📈 Results

The model successfully generates descriptive captions for uploaded images using a CNN-LSTM architecture.

Example:

**Input**

Image of a surfer riding a wave.

**Generated Caption**

```
a man in a yellow surfboard is riding a surfboard in the ocean
```

---

## 🔮 Future Improvements

- Beam Search Decoding
- Attention Mechanism
- ResNet50 / EfficientNet Feature Extractor
- Vision Transformer (ViT)
- BLIP / Transformer-based Captioning
- Deploy using Hugging Face Spaces
- Docker Support

---

## 📚 Skills Demonstrated

- Computer Vision
- Deep Learning
- Natural Language Processing
- TensorFlow/Keras
- CNN
- LSTM
- Transfer Learning
- Feature Extraction
- Sequence Modeling
- Streamlit
- Python

---

## 👨‍💻 Author

**Ashish Kumar**

B.Sc. Artificial Intelligence & Machine Learning

GitHub: https://github.com/Ashishkumar217

LinkedIn: *(Add your LinkedIn profile here)*

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.