# Parkinson's Disease Diagnosis Using Gait Metrics

## Overview

This project applies **deep learning** techniques to diagnose **Parkinson's Disease (PD)** by analyzing gait metrics, specifically focusing on **Step-Time Intervals** and **Step-Force**. Parkinson's Disease affects millions globally, but early diagnosis remains a challenge due to a lack of specific tests. By leveraging gait data, this project aims to build a robust diagnostic model.

## Key Technologies

- **Deep Learning**: Implemented using **TensorFlow/Keras** for modeling the neural networks.
- **Convolutional Neural Networks (CNN)**: To analyze time-series gait data effectively.
- **Data Processing**: **Pandas** for handling datasets, and **Scikit-learn** for normalization and model evaluation.

## Highlight: Custom Deep Learning Model

The core of this project is a **2D Convolutional Neural Network (CNN)** designed to work with **gait force data**. Here’s a glimpse of the custom CNN architecture:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatt  # Only showcasing the initial 500 characters of the Python code for brevity.
```

This model was designed with the following structure:
- **24 filters** with **7x7 kernel size**.
- **Input shape** tailored for gait data from Parkinson's disease datasets.
- **Dropout layers** for regularization to avoid overfitting on the small dataset.

## Key Features of the Model

- **Step-Time Interval vs Step-Force**: A comparison of two major metrics to diagnose PD.
- **Custom CNN Architecture**: Tailored specifically for time-series data classification.
- **Accuracy**: Achieved **80% accuracy** with Gait Force data on the test set, outperforming Step-Time Interval analysis.

### Example Code: Training the CNN

Here’s an example of how the CNN is trained, showcasing a deep learning workflow:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

# Define the CNN model
model = Sequential([
    Conv2D(24, (7, 7), activation='relu', input_shape=(64, 64, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

## Data Processing and Preparation

The dataset is processed using **Pandas** and normalized for model training. The following steps highlight the data processing pipeline:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('gait_data.csv')

# Normalizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['step_force', 'step_time_interval']])

# Preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2)
```

## Model Performance

The model performed well, achieving:
- **80% accuracy** on the test set using Gait Force data.
- **60% accuracy** using Step-Time Interval data.

## Future Work

- Expand the dataset to refine the model further.
- Explore additional gait metrics and incorporate more complex neural network architectures (e.g., **LSTM** for sequence prediction).
- Compare the model with other **neurological disorders** to improve diagnostic accuracy.

## Installation and Setup

To set up and run the project locally:

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/parkinsons-gait-classifier.git
cd parkinsons-gait-classifier
```

Run the classifier:

```bash
python parkinsons.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
