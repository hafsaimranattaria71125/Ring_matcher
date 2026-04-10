# 💍 Ring Matcher

A computer vision system that matches rings from a catalog, even when the query image contains a hand.

---

## ✨ Features

- 🔍 Hand-aware ring extraction (YOLO + SAM)
- 🔄 Rotation-invariant matching
- 🎯 Multi-level similarity: Exact / Very Similar / Similar
- 🖼️ Optional image enhancement (Real-ESRGAN)
- 📚 Fast catalog indexing with embeddings

---

## 🛠️ Tech Stack

Python, PyTorch, OpenCV, NumPy, PIL,  
YOLO (Ultralytics), SAM, CLIP, Real-ESRGAN,  
scikit-learn, matplotlib

---

## ⚙️ Setup

```bash
pip install torch torchvision opencv-python pillow numpy matplotlib scikit-learn
pip install ultralytics
pip install git+https://github.com/openai/CLIP.git
pip install segment-anything
pip install basicsr facexlib gfpgan realesrgan
