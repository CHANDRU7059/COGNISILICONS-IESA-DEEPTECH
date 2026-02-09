# sem-wefer
This is a professional **README.md** tailored for your GitHub repository. It includes project details, setup instructions, and deployment information.
# dataset link 
https://drive.google.com/file/d/1P7dm6TP0T9_0jyYLPFwsAa1FQcpwppXS/view?usp=drivesdk


***

# Semiconductor Wafer Defect Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005C92?style=flat&logo=onnx&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

An end-to-end Deep Learning pipeline to identify and classify defects in semiconductor wafers. This project uses Transfer Learning with **MobileNetV2** for high accuracy and efficiency, suitable for industrial visual inspection.

## 🚀 Key Features
*   **Fully Automated Pipeline:** Handles dataset splitting (Train/Val/Test) automatically from a single ZIP upload.
*   **Optimized Architecture:** Uses MobileNetV2 for a balance between high accuracy and fast inference.
*   **Mixed Precision Training:** Uses `torch.cuda.amp` for faster training on NVIDIA GPUs.
*   **Production Ready:** Exports the trained model to **ONNX format** with dynamic axes for seamless deployment on edge devices or web servers.
*   **Visualization:** Includes confusion matrix generation and a single-image inference tester.

## 📂 Dataset Structure
The script expects a `.zip` file containing a `train` folder with subfolders named after each defect class:

```text
your_data.zip
└── train/
    ├── bridge/
    ├── crack/
    ├── pinhole/
    └── normal/
```
*The script will automatically create a 20% Validation and 10% Test split from your training data.*

## 🛠️ Installation & Usage

### 1. Run in Google Colab
1.  Upload the `.ipynb` file to your Google Drive.
2.  Open it with **Google Colab**.
3.  Set Runtime Type to **GPU** (Runtime > Change runtime type > GPU).
4.  Run all cells.

### 2. Training the Model
*   When prompted, upload your dataset ZIP file.
*   The model will train for a default of 20 epochs (adjustable in the `Config` cell).
*   The best model weights are saved automatically as `best.pt`.

### 3. Export to ONNX
After training, the final cell automatically converts the PyTorch model to `wafer_defect_model.onnx`. This file is optimized for:
*   **ONNX Runtime** (C++, Python, C#)
*   **OpenVINO** (Intel CPUs/VPUs)
*   **TensorRT** (NVIDIA GPUs)

## 📊 Model Performance
The model architecture includes:
- **Base:** MobileNetV2 (Pre-trained on ImageNet)
- **Head:** Dropout (0.2) -> Linear (256) -> ReLU -> Linear (Num Classes)
- **Optimizer:** AdamW with Cosine Annealing Learning Rate Scheduler.

## 🖥️ Requirements
```bash
pip install torch torchvision onnx onnxruntime onnxscript scikit-learn seaborn tqdm pillow
```

## 📜 Deployment Note
The exported `wafer_defect_model.onnx` uses the following input specifications:
- **Input Name:** `input`
- **Input Shape:** `[Batch_Size, 3, 224, 224]`
- **Normalization:** Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`

## 🤝 Contributing
Feel free to fork this project, open issues, or submit pull requests to improve the classification accuracy or add support for new architectures.

