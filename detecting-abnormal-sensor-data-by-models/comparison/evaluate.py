import pandas as pd
import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"../data/sensor_data_training.csv")
X = data[['Temperature_C', 'Humidity_Percentage', 'SmokeLevel_PPM', 'CarbonMonoxideLevel_PPM']]
y = np.array([1 if i == 'Outlier' else 0 for i in data['Outlier']])

# Load models
with open('../old_model/model.pkl', 'rb') as f:
    iso_forest = pickle.load(f)

# Predictions from Isolation Forest
iso_preds = iso_forest.predict(X)
iso_preds = [1 if pred == 1 else 0 for pred in iso_preds]

# Evaluate Isolation Forest model
iso_accuracy = accuracy_score(y, iso_preds)
iso_conf_matrix = confusion_matrix(y, iso_preds)
iso_class_report = classification_report(y, iso_preds)

print("Isolation Forest Model Evaluation:")
print(f"Accuracy: {iso_accuracy}")
print(f"Confusion Matrix:\n{iso_conf_matrix}")
print(f"Classification Report:\n{iso_class_report}")

# Load PyTorch model
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

pytorch_model = AnomalyDetector(input_dim=X.shape[1])
pytorch_model.load_state_dict(torch.load('../new_model/model_weights.pth'))
pytorch_model.eval()

# Convert data to tensor for PyTorch model
X_tensor = torch.tensor(X.values, dtype=torch.float32)

# Predict using PyTorch model
with torch.no_grad():
    output = pytorch_model(X_tensor)
    pytorch_preds = (output > 0.5).float()

# Evaluate PyTorch model
pytorch_accuracy = accuracy_score(y, pytorch_preds.numpy())
pytorch_conf_matrix = confusion_matrix(y, pytorch_preds.numpy())
pytorch_class_report = classification_report(y, pytorch_preds.numpy())

print("\nPyTorch Model Evaluation:")
print(f"Accuracy: {pytorch_accuracy}")
print(f"Confusion Matrix:\n{pytorch_conf_matrix}")
print(f"Classification Report:\n{pytorch_class_report}")

# Save results
results = {
    'Model': ['Isolation Forest', 'PyTorch Model'],
    'Accuracy': [iso_accuracy, pytorch_accuracy],
}

results_df = pd.DataFrame(results)
results_df.to_csv('results_old_vs_new.csv', index=False)

# Optionally, plot comparison of confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].matshow(iso_conf_matrix, cmap='Blues')
ax[0].set_title('Isolation Forest Confusion Matrix')
ax[1].matshow(pytorch_conf_matrix, cmap='Blues')
ax[1].set_title('PyTorch Model Confusion Matrix')

plt.show()
