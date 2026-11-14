# 🛰️ EarthVision AI: Urban Expansion Monitoring

EarthVision AI is an end-to-end Deep Learning computer vision system designed to detect environmental changes and urban expansion using multi-temporal satellite imagery.

This project was built to answer the research question: *How effectively can deep learning-based computer vision identify urban expansion and environmental changes using multi-temporal satellite imagery?*

## 🧠 System Architecture

The system uses a two-stage deep learning pipeline:

1. **Land Classification (U-Net):** Trained on the EuroSAT dataset to classify pixels into 4 categories (Buildings, Vegetation, Roads, Water).
2. **Change Detection (Siamese U-Net):** Trained on the LEVIR-CD dataset. Two images (Before and After) pass through a shared encoder, and their features are concatenated to predict a binary change mask.

The outputs are fed into a Streamlit web dashboard which calculates the percentage of building growth and visualizes the changes.

## 📊 Datasets Used

- **EuroSAT:** 27,000 labeled Sentinel-2 patches for land use classification.
- **LEVIR-CD:** 637 high-resolution bi-temporal image pairs for building change detection.

---

## 🚀 How to Run Locally

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10+
- Git

### Step 1: Clone the Repository

Open your terminal (Command Prompt, PowerShell, or Terminal) and run:

```bash
git clone https://github.com/ayeshaDeveloper/EarthVision-AI.git
cd EarthVision-AI
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision segmentation-models-pytorch streamlit opencv-python rasterio
```

### Step 3: Verify the Pre-trained Models

The pre-trained model weights are already included in this repository to make setup easy.

Ensure that the `models/` folder contains the following two files:

```text
models/
├── unet_landcover.pth
└── siamese_cd.pth
```

### Step 4: Run the Streamlit Dashboard

Start the app using the following command:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at:

http://localhost:8501

1. Upload a "Before" image on the left.
2. Upload an "After" image on the right.
3. Click **Analyze Changes** to see the AI detect urban expansion!
