import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np
import joblib
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load training and testing datasets
train_data = pd.read_csv('cleaned_BBC.csv')  # Training dataset
test_data = pd.read_csv('cleaned_test.csv')  # Test dataset

# Load the test labels from test_labels.csv
test_labels = pd.read_csv('test_labels.csv')  # Test labels for accuracy evaluation

# Define feature and target columns
X = train_data['words'] # Words column (full dataset)
y = train_data['category'] # Target column (full dataset)

# Encode target column 'y' to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_test = label_encoder.transform(test_labels['category'])
# print("label encoding order:", label_encoder.classes_) ['business' 'entertainment' 'politics' 'sport' 'tech']


# Open a file to write the results
with open('trainResults.txt', 'w') as f:
    try:
        # Define the feature and target columns
        X_train = train_data['words']  # Text column (training set)
        y_train = train_data['category']  # Target column (training set)
        
        X_test = test_data['words']  # Text column (test set)
        y_test = test_labels['category'] # True labels for the test set

        # Encode the target column 'y' to numeric labels using LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)  # Ensure test labels are encoded similarly

        # Log the unique labels in y_train and y_test to help with debugging
        f.write(f"Unique classes in y_train: {np.unique(y_train)}\n")
        f.write(f"Unique classes in y_test: {np.unique(y_test)}\n")

        # Convert the text data to numerical features using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Convert the training and test sets to dense arrays (for compatibility with models)
        X_train_tfidf = X_train_tfidf.toarray()
        X_test_tfidf = X_test_tfidf.toarray()

        # Normalize the data
        scaler = StandardScaler()
        X_train_tfidf = scaler.fit_transform(X_train_tfidf)
        X_test_tfidf = scaler.transform(X_test_tfidf)

        # ------------------- XGBoost Model -------------------
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train_tfidf, y_train)

        # Make predictions on the test set
        y_pred_xgb = xgb_model.predict(X_test_tfidf)

        # Calculate accuracy of the predictions using test labels
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        f.write(f'XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%\n')
        # Print classification report for XGBoost
        class_names = label_encoder.classes_
        xgb_report = classification_report(y_test, y_pred_xgb, target_names=class_names)
        f.write(f"\nXGBoost Classification Report:\n{xgb_report}\n")

        # Save XGBoost model
        joblib.dump(xgb_model, 'xgb_model.pkl')

        # ------------------- SVM Model -------------------
        # Linear kernel SVM
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train_tfidf, y_train)

        # Make predictions with SVM
        y_pred_svm = svm_model.predict(X_test_tfidf)

        # Calculate accuracy of the SVM model
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        f.write(f'SVM Model Accuracy: {accuracy_svm * 100:.2f}%\n')

        # Print classification report for SVM
        svm_report = classification_report(y_test, y_pred_svm, target_names=class_names)
        f.write(f"\nSVM Classification Report:\n{svm_report}\n")

        # Save SVM model
        joblib.dump(svm_model, 'svm_model.pkl')

        # Save the TF-IDF vectorizer
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

        # Save the scaler
        joblib.dump(scaler, 'scaler.pkl')
    except Exception as e:
        f.write(f"An error occurred: {str(e)}\n")