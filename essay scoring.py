# %%
# Importing necessary libraries and packages

import os
import re
import zipfile

import nltk
import numpy as np
import pandas as pd
import requests
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    cohen_kappa_score,
    explained_variance_score,
    mean_squared_error,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

nltk.download("punkt")
nltk.download("stopwords")


# %%
def download_dataset_file(url, destination_path):
    """
    Downloads a file from the specified URL and saves it to the given path.
    """
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


# %%
# Function to unzip the dataset file
def extract_zip_file(zip_file_path=None):
    """
    Extracts the contents of a zip file into the 'Dataset' directory.
    """
    try:
        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall("Dataset")
            print("Extraction completed successfully.")
    except zipfile.BadZipFile:
        print("Error: Invalid zip file provided.")


extract_zip_file("Dataset/essay_scorer_dataset.zip")

# %%
# Load dataset
training_data = pd.read_csv("Dataset/training_set.tsv", sep="\t", encoding="ISO-8859-1")

# Extract dependent variable
target_scores = training_data["domain1_score"]
essays_df = training_data.loc[:, ["essay_id", "essay_set", "essay", "domain1_score"]]

# %%
# Tokenize words from essays after cleaning text by removing non-alphabetic characters,
# converting to lowercase, and removing stopwords


def tokenize_words(essay_text):
    """
    Cleans and tokenizes words from the provided essay text.
    """
    clean_text = re.sub("[^a-zA-Z]", " ", essay_text)
    words = clean_text.lower().split()
    stop_words = set(stopwords.words("english"))
    filtered_words = [w for w in words if w not in stop_words]
    return filtered_words


# %%
# Tokenize sentences from essays and subsequently tokenize words within those sentences


def tokenize_sentences(essay_text):
    """
    Tokenizes sentences from the provided essay text, and then tokenizes words within those sentences.
    """
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = tokenizer.tokenize(essay_text.strip())
    tokenized_sentences = [
        tokenize_words(sentence) for sentence in sentences if len(sentence) > 0
    ]
    return tokenized_sentences


# %%
# Generate a feature vector for the provided words


def generate_feature_vector(words, model, num_features):
    """
    Generates a feature vector by averaging word vectors from the provided word list.
    """
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


# %%
# Generate average feature vectors for a list of essays


def generate_avg_feature_vectors(essays, model, num_features):
    """
    Generates average feature vectors for each essay in the list of essays.
    """
    essay_feature_vectors = np.zeros((len(essays), num_features), dtype="float32")

    for i, essay_text in enumerate(essays):
        essay_feature_vectors[i] = generate_feature_vector(
            essay_text, model, num_features
        )

    return essay_feature_vectors


# %%
def build_lstm_model():
    """
    Builds and compiles an LSTM model for essay scoring.
    """
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(
            300,
            dropout=0.4,
            recurrent_dropout=0.4,
            input_shape=[1, 300],
            return_sequences=True,
        )
    )
    lstm_model.add(LSTM(64, recurrent_dropout=0.4))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(1, activation="relu"))

    lstm_model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"])
    lstm_model.summary()

    return lstm_model


# %%
# Applying k-fold cross-validation

cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
evaluation_results = []
predicted_scores = []

fold_counter = 1
for train_indices, test_indices in cross_validator.split(essays_df):

    print("\n------------ Fold {} ------------\n".format(fold_counter))
    train_set, test_set = essays_df.iloc[train_indices], essays_df.iloc[test_indices]
    y_train, y_test = (
        target_scores.iloc[train_indices],
        target_scores.iloc[test_indices],
    )

    train_essays = train_set["essay"]
    test_essays = test_set["essay"]

    # Tokenize sentences from training essays
    all_sentences = []
    for essay in train_essays:
        all_sentences += tokenize_sentences(essay)

    # Word2Vec model parameters
    vector_size = 300
    min_word_count = 40
    num_workers = 4
    context_window = 10
    downsampling = 1e-3

    print("Training Word2Vec model...")
    word2vec_model = Word2Vec(
        all_sentences,
        workers=num_workers,
        vector_size=vector_size,
        min_count=min_word_count,
        window=context_window,
        sample=downsampling,
    )

    word2vec_model.init_sims(replace=True)
    word2vec_model.wv.save_word2vec_format("word2vec_model.bin", binary=True)

    # Generate feature vectors for training and testing sets
    clean_train_essays = [tokenize_words(essay) for essay in train_essays]
    train_data_vectors = generate_avg_feature_vectors(
        clean_train_essays, word2vec_model, vector_size
    )

    clean_test_essays = [tokenize_words(essay) for essay in test_essays]
    test_data_vectors = generate_avg_feature_vectors(
        clean_test_essays, word2vec_model, vector_size
    )

    train_data_vectors = np.reshape(
        train_data_vectors,
        (train_data_vectors.shape[0], 1, train_data_vectors.shape[1]),
    )
    test_data_vectors = np.reshape(
        test_data_vectors, (test_data_vectors.shape[0], 1, test_data_vectors.shape[1])
    )

    lstm_model = build_lstm_model()
    lstm_model.fit(train_data_vectors, y_train, batch_size=64, epochs=50)
    predicted_y = lstm_model.predict(test_data_vectors)

    # Round predicted values to nearest integer
    predicted_y = np.around(predicted_y)

    """Evaluation metrics used:
    1. Mean squared error
    2. Explained variance score
    3. Cohen's kappa score
    Expected results: Minimum error, maximum variance, and maximum kappa score."""

    # Mean squared error
    print(
        "Mean squared error: {0:.2f}".format(
            mean_squared_error(y_test.values, predicted_y)
        )
    )

    # Explained variance score
    print(
        "Explained variance score: {0:.2f}".format(
            explained_variance_score(y_test.values, predicted_y)
        )
    )

    # Cohen's kappa score
    kappa_score = cohen_kappa_score(y_test.values, predicted_y, weights="quadratic")
    print("Cohen's Kappa Score: {0:.2f}".format(kappa_score))
    evaluation_results.append(kappa_score)

    fold_counter += 1

# %%
print(
    "Average Cohen's Kappa score after 5-fold cross-validation: ",
    np.around(np.mean(evaluation_results), decimals=2),
)

# %%
# Splitting dataset into training and test set and generating word embeddings for other models

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    training_data, target_scores, test_size=0.25
)

train_essays_split = X_train_split["essay"]
test_essays_split = X_test_split["essay"]

sentences_split = []

for essay_split in train_essays_split:
    sentences_split += tokenize_sentences(essay_split)

# Initializing variables for Word2Vec model
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print("Training Word2Vec Model...")
word2vec_model = Word2Vec(
    sentences_split,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=context,
    sample=downsampling,
)

word2vec_model.init_sims(replace=True)
word2vec_model.wv.save_word2vec_format("word2vec_model.bin", binary=True)

clean_train_essays_split = []

# Generate training and testing data word vectors
for essay_text_split in train_essays_split:
    clean_train_essays_split.append(tokenize_words(essay_text_split))
trainDataVecs_split = generate_avg_feature_vectors(
    clean_train_essays_split, word2vec_model, num_features
)

clean_test_essays_split = []
for essay_text_split in test_essays_split:
    clean_test_essays_split.append(tokenize_words(essay_text_split))
testDataVecs_split = generate_avg_feature_vectors(
    clean_test_essays_split, word2vec_model, num_features
)

trainDataVecs_split = np.array(trainDataVecs_split)
testDataVecs_split = np.array(testDataVecs_split)

# %%
# Generating scores using Linear Regression Model

linear_regressor_split = LinearRegression()

linear_regressor_split.fit(trainDataVecs_split, y_train_split)

y_pred_split = linear_regressor_split.predict(testDataVecs_split)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test_split, y_pred_split))

# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % explained_variance_score(y_test_split, y_pred_split))

# Cohen's kappa score
print(
    "Kappa Score: {0:.2f}".format(
        cohen_kappa_score(
            y_test_split.values, np.around(y_pred_split), weights="quadratic"
        )
    )
)

# %%
# Generating scores using Gradient Boosting Regressor

gbr_split = ensemble.GradientBoostingRegressor(
    alpha=0.9,
    criterion="friedman_mse",
    init=None,
    learning_rate=0.1,
    loss="squared_error",
    max_depth=2,
    max_features=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=1000,
    random_state=None,
    subsample=1.0,
    verbose=0,
    warm_start=False,
)
gbr_split.fit(trainDataVecs_split, y_train_split)
y_pred_split = gbr_split.predict(testDataVecs_split)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test_split, y_pred_split))

# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % explained_variance_score(y_test_split, y_pred_split))

# Cohen's kappa score
print(
    "Kappa Score: {0:.2f}".format(
        cohen_kappa_score(
            y_test_split.values, np.around(y_pred_split), weights="quadratic"
        )
    )
)

# %%
# Generating scores using Support Vector Regression (SVR)

svr_split = SVR(
    C=100,
    cache_size=200,
    coef0=0.0,
    degree=3,
    epsilon=0.1,
    gamma=0.1,
    kernel="rbf",
    max_iter=-1,
    shrinking=True,
    tol=0.001,
    verbose=False,
)
svr_split.fit(trainDataVecs_split, y_train_split)
y_pred_split = svr_split.predict(testDataVecs_split)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test_split, y_pred_split))

# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % explained_variance_score(y_test_split, y_pred_split))

# Cohen's kappa score
print(
    "Kappa Score: {0:.2f}".format(
        cohen_kappa_score(
            y_test_split.values, np.around(y_pred_split), weights="quadratic"
        )
    )
)

# %%
# As LSTM outperforms all other models, using it for predicting the scores for the final dataset
validation_set = pd.read_csv("Dataset/valid_set.tsv", sep="\t", encoding="ISO-8859-1")

# %%
validation_set = validation_set.drop(["domain2_predictionid"], axis=1)

# %%
valid_test_essays = validation_set["essay"]

# %%
sentences_valid = []

for valid_essay in valid_test_essays:
    sentences_valid += tokenize_sentences(valid_essay)

print("Training Word2Vec Model...")
word2vec_model = Word2Vec(
    sentences_valid,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=context,
    sample=downsampling,
)

word2vec_model.init_sims(replace=True)
word2vec_model.wv.save_word2vec_format("word2vec_model.bin", binary=True)

clean_valid_test_essays = []

# Generate testing data word vectors
for essay_text_valid in valid_test_essays:
    clean_valid_test_essays.append(tokenize_words(essay_text_valid))
valid_testDataVecs = generate_avg_feature_vectors(
    clean_valid_test_essays, word2vec_model, num_features
)

valid_testDataVecs = np.array(valid_testDataVecs)
# Reshaping test vectors to 3 dimensions (1 represents one timestep)
valid_testDataVecs = np.reshape(
    valid_testDataVecs, (valid_testDataVecs.shape[0], 1, valid_testDataVecs.shape[1])
)

predicted_scores_valid = lstm_model.predict(valid_testDataVecs)

# Round predicted scores to the nearest integer
predicted_scores_valid = np.around(predicted_scores_valid)

# %%
submission = validation_set.drop(["essay"], axis=1)

# %%
predicted_score_series = pd.Series(
    [score for sublist in predicted_scores_valid for score in sublist]
)

# %%
submission = (
    pd.concat([submission, predicted_score_series], axis=1)
    .rename(columns={0: "predicted_score"})
    .iloc[:, [2, 0, 1, 3]]
)
submission.to_excel("Submission.xlsx", index=False)
