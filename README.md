## Table of Contents

- [Automated Essay Scoring Model](#automated-essay-scoring-model)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Using the Model](#using-the-model)
    - [Scoring a Custom Essay](#scoring-a-custom-essay)
    - [Exporting and Loading Trained Models](#exporting-and-loading-trained-models)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Results](#results)
  - [License](#license)


# Automated Essay Scoring Model

This project aims to build a machine learning model that automatically scores essays based on a provided dataset. The model is implemented using a combination of NLP techniques, machine learning algorithms, and deep learning models.

## Features

- **LSTM Model** for essay scoring.
- **Word2Vec** for word embeddings.
- **Support for multiple regression models** (Linear Regression, Gradient Boosting Regressor, Support Vector Regression).
- **K-Fold Cross-Validation** for model evaluation.
- **Exporting and Loading Trained Models** for future predictions.

## Dataset

The dataset required for this project will be automatically downloaded and extracted when the script or notebook is run. It contains essay texts along with their corresponding scores.

## Getting Started

### Prerequisites

Ensure that you have Python 3.11.9 installed along with the necessary libraries. You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/a-sajjad72/CodeAlpha_Automated_Essay_Scoring.git
   cd CodeAlpha_Automated_Essay_Scoring
   ```

2. **Download and Extract Dataset:**

   The dataset will be automatically downloaded and extracted when you run the notebook or script. You do not need to manually download the data. If you want to download the dataset manually, you can use the following code snippet:

   ```python
   import os
   import requests
   import zipfile

   def download_dataset_file(url, destination_path):
       if not os.path.exists(destination_path):
           response = requests.get(url)
           with open(destination_path, "wb") as file:
               file.write(response.content)

   # Create a directory for the dataset
   os.makedirs("Dataset", exist_ok=True)

   download_dataset_file(
       "https://www.dropbox.com/scl/fi/0s4skjwowniwy1pn3uwaz/essay_scorer_dataset.zip?rlkey=uhcmx6z82llbkc7hs24ww97fl&st=l55tyxi3&dl=1",
       "Dataset/essay_scorer_dataset.zip",
   )

   def extract_zip_file(zip_file_path=None):
       with zipfile.ZipFile(zip_file_path, "r") as z:
           z.extractall("Dataset")
           print("Extraction completed successfully.")

   extract_zip_file("Dataset/essay_scorer_dataset.zip")
   ```

3. **Run the Jupyter Notebook:**

   Launch the Jupyter notebook to interact with the model and run the training code.

   ```bash
   jupyter notebook essay_scoring.ipynb
   ```

4. **Run the Python Script:**

   Alternatively, you can execute the Python script directly.

   ```bash
   python essay_scoring.py
   ```

## Using the Model

### Scoring a Custom Essay

To score a custom essay using the trained LSTM model, you can use the following code snippet:

```python
# Load necessary libraries and models
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np

# Load the LSTM model
lstm_model = load_model('lstm_model.h5')

# Load the Word2Vec model
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Function to tokenize and clean the essay
def tokenize_words(essay_text):
    import re
    from nltk.corpus import stopwords
    clean_text = re.sub("[^a-zA-Z]", " ", essay_text)
    words = clean_text.lower().split()
    stop_words = set(stopwords.words("english"))
    filtered_words = [w for w in words if w not in stop_words]
    return filtered_words

# Function to generate feature vector
def generate_feature_vector(words, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0.0
    model_vocab = set(model.wv.index_to_key)

    for word in words:
        if word in model_vocab:
            num_words += 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)

    return feature_vector

# Custom essay text
custom_essay = "Your essay text goes here."

# Tokenize and vectorize the custom essay
tokenized_essay = tokenize_words(custom_essay)
vectorized_essay = generate_feature_vector(tokenized_essay, word2vec_model, 300)
vectorized_essay = np.reshape(vectorized_essay, (1, 1, 300))

# Predict the score
predicted_score = lstm_model.predict(vectorized_essay)
predicted_score = np.around(predicted_score)

print("Predicted Score:", predicted_score)
```

### Exporting and Loading Trained Models

This project trains multiple models. Here’s how to export and load each of them:

#### LSTM Model

- **Exporting:**

  ```python
  # Save the trained LSTM model
  lstm_model.save('lstm_model.h5')
  ```

- **Loading:**

  ```python
  from tensorflow.keras.models import load_model

  # Load the LSTM model
  lstm_model = load_model('lstm_model.h5')
  ```

#### Linear Regression Model

- **Exporting:**

  ```python
  import joblib

  # Save the trained Linear Regression model
  joblib.dump(linear_regressor_split, 'linear_regressor_model.pkl')
  ```

- **Loading:**

  ```python
  import joblib

  # Load the Linear Regression model
  linear_regressor_split = joblib.load('linear_regressor_model.pkl')
  ```

#### Gradient Boosting Regressor

- **Exporting:**

  ```python
  # Save the trained Gradient Boosting Regressor model
  joblib.dump(gbr_split, 'gbr_model.pkl')
  ```

- **Loading:**

  ```python
  # Load the Gradient Boosting Regressor model
  gbr_split = joblib.load('gbr_model.pkl')
  ```

#### Support Vector Regression (SVR)

- **Exporting:**

  ```python
  # Save the trained SVR model
  joblib.dump(svr_split, 'svr_model.pkl')
  ```

- **Loading:**

  ```python
  # Load the SVR model
  svr_split = joblib.load('svr_model.pkl')
  ```

## Evaluation Metrics

The model's performance is evaluated using the following metrics:

- **Mean Squared Error (MSE)**
- **Explained Variance Score**
- **Cohen's Kappa Score**

### Results

Here are the evaluation results for each model:

#### LSTM Model (After 5-Fold Cross-Validation)

- **Average Mean Squared Error:** 6.41
- **Average Explained Variance Score:** 0.92
- **Average Cohen's Kappa Score:** 0.96

#### Linear Regression Model

- **Mean Squared Error:** 19.76
- **Explained Variance Score:** 0.75
- **Cohen's Kappa Score:** 0.86

#### Gradient Boosting Regressor

- **Mean Squared Error:** 6.83
- **Explained Variance Score:** 0.91
- **Cohen's Kappa Score:** 0.96

#### Support Vector Regression (SVR)

- **Mean Squared Error:** 28.94
- **Explained Variance Score:** 0.64
- **Cohen's Kappa Score:** 0.74

**Conclusion:** The LSTM model outperforms the other models in terms of all evaluation metrics, making it the preferred choice for this essay scoring task.

## License

This project is licensed under the MIT License - see the [MIT License](LICENSE) file for details.
