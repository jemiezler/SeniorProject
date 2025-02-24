# 🚀 Background Segmentation API (FastAPI + DeepLabV3+)

This is a **FastAPI-based segmentation API** that uses a pre-trained **DeepLabV3+** model for background segmentation.

---

## **🛠 Installation Guide**

### **1️⃣ Create a Virtual Environment**
Before installing dependencies, create and activate a **virtual environment**:

#### **For Linux/macOS:**
```bash
cd src  # Navigate to the FastAPI project directory
python3 -m venv venv
source venv/bin/activate  # Activate virtual environment
```

#### **For Windows:**
```bash
cd src
python -m venv venv
venv\Scripts\activate  # Activate virtual environment
```

### **2️⃣ Install Dependencies**
After activating the virtual environment, install the required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
---
## **🚀 Running the API**
To start the API, run:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```