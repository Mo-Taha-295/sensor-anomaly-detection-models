# sensor-anomaly-detection-models
Comparative analysis of  models for detecting abnormal sensor data in IoT-based fire detection systems.
# Detecting Abnormal Sensor Data by Models

This project presents a comparative study between two models for detecting abnormal (outlier) sensor values in IoT-based fire detection systems. The goal is to identify unusual readings from sensors such as temperature, humidity, smoke, and CO levels that may indicate critical events like fire or sensor malfunction.

## Models Compared

### 🔹 Model 1: Isolation Forest (Scikit-learn)
A classic unsupervised learning algorithm used for anomaly detection. This model was trained on four sensor features:
- Temperature_C
- Humidity_Percentage
- SmokeLevel_PPM
- CarbonMonoxideLevel_PPM

**Results:**
- Accuracy: **99.75%**

---

### 🔹 Model 2: Custom Neural Network (PyTorch)
A simple feed-forward neural network trained in a supervised manner using the output of the Isolation Forest as labels. This approach simulates how a learned model could mimic or extend the unsupervised method.

**Architecture:**
- Input Layer → 10 neurons → 5 neurons → 1 output neuron (sigmoid)

**Training Details:**
- Loss: Binary Cross Entropy (BCELoss)
- Optimizer: Adam
- Epochs: 100

**Results:**
- Accuracy: **100%**
**Note:** While the PyTorch model achieved 100% accuracy on the current dataset, this result should be interpreted with caution. 
Perfect accuracy is often a sign of overfitting, especially when the dataset is limited in size or generated using labels from another model. In real-world scenarios, 
as the dataset grows in size and complexity, 
the model's performance may decrease, revealing more realistic behavior. Additional evaluation on unseen or real-time data is recommended.


---

## Dataset

The dataset contains readings from four environmental sensors used to detect potential fire-related anomalies:

- `Temperature_C`
- `Humidity_Percentage`
- `SmokeLevel_PPM`
- `CarbonMonoxideLevel_PPM`

In addition to these features, a fifth column — `Outlier` — was generated using the Isolation Forest model. This column labels each data point as either:
- `'Normal'` (non-anomalous)
- `'Outlier'` (anomalous)

This `Outlier` column served as the target label for the supervised training of the PyTorch neural network model.

🔸 **Note:** While using Isolation Forest to generate labels provided a useful baseline for model comparison, relying on machine-generated labels can introduce bias. For real-world deployment and reliable performance, manually annotated or real-world labeled data is strongly recommended.


---

## Future Work
- Train the PyTorch model on manually labeled data (instead of Isolation Forest labels)
- Test on real-time streaming data from ESP32 sensor module
- Deploy as a lightweight anomaly detection service on edge devices

---


MIT License

Copyright (c) @Mohamed Taha Eslayed (2025)





