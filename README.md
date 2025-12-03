# 🛠️ Surface Defect Detection (Quality vs Non-Quality)

This project is an end-to-end **surface defect classification system** that automatically detects whether industrial products are **Quality** or **Non-Quality** by analyzing their surface patterns.



## 📌 1. Project Overview

The system uses computer vision + deep learning to identify defects on metal/plastic/industrial surfaces.

### Model Output Classes:

* **Quality** → normal surface
* **Non-Quality** → defective surface (scratches, cracks, dents, noise patterns, etc.)

This automated system can replace slow and error-prone human inspection in factories.

---

## 📌 2. Dataset Structure

### **Raw Data**

```
raw_defect_data/
    quality/
        image_001.png
        ...
    non_quality/
        image_002.png
        ...
```

### **Processed Data**

(Masks created automatically by the script)

```
processed_defect_data/
    quality/
    non_quality/
```

The preprocessing step enhances possible surface defects using:

* Grayscale conversion
* Gaussian blur
* Laplacian edge detection
* Binary thresholding

---

## 📌 3. How the Model Works

1️⃣ **Image Preprocessing**
Converts raw images into binary masks to highlight defects.

2️⃣ **CNN Classification Model**
A custom Convolutional Neural Network learns defect features:

* 3× Conv + MaxPooling layers
* Dense layers
* Softmax classifier (2 classes)

3️⃣ **Prediction on New Images**
Any new image can be passed into the model, and it outputs:

* Predicted class
* Confidence percentage

---

## 📌 4. Training the Model

Run:

```bash
python detect.py
```

The script performs:

* Preprocessing
* Training
* Saving model as `defect_classifier_model.h5`
* Exporting:

  * `summary.txt`
  * `architecture.json`
  * all weights as `.npy`
  * `model.png` (if graphviz installed)

---

## 📌 5. Testing a New Image

Place your test image in the project folder and set the filename in the code:

```python
TEST_IMAGE_FILE = "image_009383.png"
```

Run:

```bash
python detect.py
```

The output example:

```
Predicted Class: NON_QUALITY
Confidence Level: 97.42%
```

---

## 📌 6. Project Files

* `detect.py` → main script
* `raw_defect_data/` → original dataset
* `processed_defect_data/` → auto-generated binary masks
* `defect_classifier_model.h5` → trained model
* `model_info/`

  * weight files
  * architecture.json
  * summary.txt
* Test image (e.g., `image_009383.png`)


## ✔️ Conclusion

This project delivers a full **AI-driven defect detection system** suitable for industrial quality control.
With automatic preprocessing, a CNN classifier, and real-time prediction, it provides a scalable solution for manufacturers.

---
