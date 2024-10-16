# Parkinson's Disease Diagnosis Using Gait Metrics

## Overview

This project focuses on diagnosing **Parkinson's Disease (PD)** by analyzing gait metrics using **machine learning models**. Parkinson’s Disease affects 7-10 million people worldwide, generally men over the age of 65. Traditional diagnosis is challenging as it relies on observing motor symptoms and assessing a patient’s medical history. This project aims to provide an **alternative diagnostic tool** by comparing **Step-Time Intervals** and **Step-Force** metrics using **Neural Networks**.

## Motivation

Currently, there are no specific tests for diagnosing Parkinson's Disease. This project explores the use of **Step-Force** as a diagnostic tool, which could revolutionize the way doctors diagnose PD by leveraging **gait force data**.

## Key Metrics

1. **Step-Time Interval** – The time between each step.
2. **Step-Force** – Force exerted during each step, recorded with multiple sensors.

These metrics are extracted from two databases:
- **Gait in Aging Database** (Step-Time Interval data)
- **Gait in Parkinson’s Disease Database** (Step-Force data)

## Objective

The goal is to determine if **Step-Force** can be as effective or more effective than **Step-Time Interval** for predicting Parkinson's Disease. By training and comparing models using these two metrics, we aim to find correlations that aid in early diagnosis.

## Data Overview

### Gait Force Data
- **Sensors**: 16 force sensors placed under each subject's feet.
- **Frequency**: 100 Hz, over a 2-minute duration.
- **Data**: Force data is captured across 8 sensors per foot and aggregated for analysis.

### Step Interval Data
- **Columns**: Timestamp (seconds), Step Interval (seconds).

## Neural Network Architecture

We use a **2D Convolutional Neural Network (CNN)** with 24 filters of size 7x7 to process the gait data. Despite the **time-series** nature of the data, a CNN was chosen instead of an RNN/LSTM due to the small dataset size and better performance on classification tasks.

- **Training Set**: 28 control examples, 28 PD examples.
- **Test Set**: 5 control examples, 5 PD examples.

### Model Performance
- **Gait Force Model**: 80% accuracy on the test set.
- **Step-Time Interval Model**: 60% accuracy on the test set.

These results indicate that **gait force** provides more detailed information than step-time intervals for diagnosing Parkinson's Disease.

## Future Directions

- Expanding the dataset to improve model accuracy and robustness.
- Exploring additional gait metrics.
- Comparing this method with other neurological diseases.
- Tracking the progression of Parkinson’s Disease over time using gait metrics.

## Why CNN and not LSTM?

Although the data is time-series, this is not a sequence prediction problem. **CNNs** perform better on smaller datasets and are more effective for time-series classification than **LSTMs**, which are prone to overfitting with limited data.

## Installation and Setup

To run the project locally, you will need the following libraries:

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

### Running the Classifier

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/parkinsons-gait-classifier.git
   ```

2. Navigate to the project directory:
   ```bash
   cd parkinsons-gait-classifier
   ```

3. Run the classifier:
   ```bash
   python parkinsons.py
   ```

## Data Processing

The script processes **Gait Force** and **Step Interval** data by normalizing the input features and training the CNN model. Results are displayed after training.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
