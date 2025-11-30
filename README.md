# 🧪 Liver Disease Classification using ResNet18

This project focuses on the automated classification of liver scan images using a deep learning model based on **ResNet18**. The goal is to assist in identifying liver conditions through medical imaging and provide a foundation for AI-assisted diagnosis.

---

## 🧠 Project Purpose

Liver-related diseases such as **Hepatocellular Carcinoma (HCC)** and **Cholangiocarcinoma (CC)** can be difficult and time-consuming to diagnose manually.  
This project demonstrates how deep learning can help:

- Automate image-based liver disease detection  
- Reduce analysis time  
- Support medical research and education  
- Serve as a prototype for clinical AI tools  

> ⚠️ **This project is for research and educational use only — not for real medical diagnosis.**

---

## 🩺 Classification Categories

The model predicts one of the following classes:

| Label | Meaning |
|-------|---------|
| NORMAL LIVER | Healthy liver scan |
| HCC | Hepatocellular Carcinoma |
| CC | Cholangiocarcinoma |

---

## 📁 Repository Contents

| File | Description |
|------|------------|
| `app.py` | Streamlit application for live image classification |
| `preprocessing.py` | Script to preprocess dataset and generate training data |
| `eda.py` | Exploratory Data Analysis to understand dataset distribution and sample images |
| `resnet18_model.pth` (optional) | Trained model weights to be used by `app.py` |

---

## 🛠️ Technologies & Libraries Used

- **Python**
- **PyTorch**
- **TorchVision**
- **Streamlit**
- **Pandas, NumPy**
- **OpenCV**
- **Matplotlib & Seaborn**

---

## 📦 Installation Guide

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/Liver-Disease-Classification-ResNet18.git
cd Liver-Disease-Classification-ResNet18
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

Place your dataset in the following structure:

```
data/liver_images/
 ├── NORMAL LIVER/
 ├── HCC/
 └── CC/
```

---

## ▶️ Running the Classifier App

```bash
streamlit run app.py
```

Streamlit will open locally at:

```
http://localhost:8501
```

Upload an image to receive a prediction.

---

## 📊 Optional Scripts

### 🔍 Run Data Analysis

```bash
python eda.py
```

### ⚙️ Run Preprocessing

```bash
python preprocessing.py
```

---

## 🚀 Future Improvements

| Feature | Status |
|--------|--------|
| Training script (`train.py`) | ⏳ Planned |
| Confusion matrix & metrics | ⏳ Planned |
| Online deployment (Streamlit Cloud/HuggingFace) | ⏳ Planned |
| Explainability (Grad-CAM) | ⏳ Planned |

---

## 🧾 License

This project is for **research and educational purposes only**.

---

## 🤝 Contributions

Contributions and improvements are welcome.  
Feel free to open issues or pull requests.

---

⭐ If you find this repository useful, please consider giving it a **star**!


