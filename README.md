
#  DeepOysterBinary

This project uses deep learning (MobileNetV2) to classify oyster images into **binary categories**: `good` or `bad`. It includes training, evaluation, and explainability techniques (like Grad-CAM) to understand the model’s predictions and error patterns.

---

##  Project Structure

```plaintext
DeepOysterBinary/
│
├── Dataset/                    # Main dataset: train/test images (augmented)
├── datasetOriginalTest/       # Clean test set without augmentations
├── DeepOysterBinary.ipynb     # Jupyter notebook with full model workflow
├── LICENSE                    # Licensing terms (optional/custom)
├── README.txt                 # Text version of this README
└── README.md                  # GitHub-formatted README
````

---

##  Model Overview

* **Model**: MobileNetV2 (transfer learning from ImageNet)
* **Task**: Binary classification (bad vs. good oysters)
* **Loss Function**: Binary Cross-Entropy with Logits
* **Optimizer**: Adam
* **Tools**: PyTorch, TorchVision, sklearn, seaborn, torchcam (Grad-CAM)

---

##  Results

* Achieved **77% accuracy** on the test set after fine-tuning.
* Significantly reduced loss from initial training phase.
* False positives analyzed using Grad-CAM visualizations to assess what regions the model focuses on.
* Applied model to clean and augmented test sets for consistency checks.

---

##  Next Steps

* Expand dataset by including **California oysters** (e.g., from Kaggle) to test model generalizability.
* Work with a project expert to **label more oyster data** or verify annotations.
* Transition from binary to **multi-class classification (5 oyster quality levels)** if more data becomes available.

---

##  References

* [MobileNetV2 - TorchVision Models](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)
* [TorchCam (Grad-CAM)](https://frgfm.github.io/torch-cam/)
* [ChatGPT (OpenAI)](https://chat.openai.com) – Used for collaborative coding and debugging support
* Custom oyster dataset from internal collection

---

##  Visual Insights

Confusion matrices, Grad-CAM overlays, and classification reports are included in the notebook for detailed evaluation.

---

## 📬 Contact

For questions, collaboration, or dataset inquiries, feel free to reach out via email or GitHub issues.

---

**License**: Custom (see `LICENSE` file)

