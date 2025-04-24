import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- STEP 1: PARTITION DATASET ----------
# The labeled training dataset is divided into 3 parts:
## train1: used to train agent1 (XGBoost)
## train2: used to train agent2 (SVM)
## train3: labels are removed to make it an unlabeled dataset


# load the dataset
data = pd.read_csv('cleaned_bbc.csv')

# ensure relevant columns are selected (category and words - preprocessed/cleaned articles)
data = data[['category', 'words']]

# split into 3 equal parts
train1, temp = train_test_split(data, test_size=2/3, random_state=42, stratify=data['category'])
train2, train3 = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['category'])

# remove labels from train3 to simulate unlabeled data
train3_unlabeled = train3.drop(columns='category')

# save the partitions for clarity
print(f"Train1: {train1.shape}, Train2: {train2.shape}, Train3 (Unlabeled): {train3_unlabeled.shape}")

# check the distribution of labels to ensure proper division
print("Train1 label distribution:")
print(train1['category'].value_counts())
print("Train2 label distribution:")
print(train2['category'].value_counts())
print("Train3 label distribution (before dropping labels):")
print(train3['category'].value_counts())



# ---------- STEP 2: INITIAL TRAINING ----------
# train each agent independently:
## agent1 (XGBoost) is trained using train1
## agent2 (SVM) is trained using train2
## both agents are evaluated against the testing dataset
## stats are outputted


# load testing data
test_data = pd.read_csv('cleaned_test.csv')
test_labels = pd.read_csv('test_labels.csv')

# initialize vectorizer + label encoder
vectorizer = TfidfVectorizer(max_features=5000)
label_encoder = LabelEncoder()

# fit vectorizer on data
vectorizer.fit(pd.concat([train1['words'], train2['words'], test_data['words']], axis=0))

# encode labels w/ LabelEncoder
label_encoder.fit(data['category'])

# extract features + labels for train1 and train2
X_train1 = vectorizer.transform(train1['words']).toarray()
y_train1 = label_encoder.transform(train1['category'])

X_train2 = vectorizer.transform(train2['words']).toarray()
y_train2 = label_encoder.transform(train2['category'])

# extract features + labels for testing dataset
X_test = vectorizer.transform(test_data['words']).toarray()
y_test = label_encoder.transform(test_labels['category'])

# scale features
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_train2 = scaler.transform(X_train2)
X_test = scaler.transform(X_test)

# train agent 1 (XGBoost)
agent_1 = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss')
agent_1.fit(X_train1, y_train1)

# train agent 2 (SVM)
agent_2 = SVC(probability=True, kernel='linear')
agent_2.fit(X_train2, y_train2)

# evaluate agents on test dataset
# agent 1 (XGBoost)
y_pred_agent1 = agent_1.predict(X_test)
accuracy_agent1 = accuracy_score(y_test, y_pred_agent1) * 100
print("Agent 1 (XGBoost) Evaluation on Test Dataset:")
print(f"Accuracy: {accuracy_agent1:.2f}%")
print(classification_report(y_test, y_pred_agent1, target_names=label_encoder.classes_))

# agent 2 (SVM)
y_pred_agent2 = agent_2.predict(X_test)
accuracy_agent2 = accuracy_score(y_test, y_pred_agent2) * 100
print("Agent 2 (SVM) Evaluation on Test Dataset:")
print(f"Accuracy: {accuracy_agent2:.2f}%")
print(classification_report(y_test, y_pred_agent2, target_names=label_encoder.classes_))

# save models and scalers
joblib.dump(agent_1, "agent_1_xgb.pkl")
joblib.dump(agent_2, "agent_2_svm.pkl")
joblib.dump(scaler, "scaler.pkl")



# ---------- STEP 3: MUTUAL LEARNING ----------
# 1. prediction phase:
## both agents make predictions for train3
## confidence scores (posterior probabilities) are recorded
# 2. relabeling phase:
## each data point in train3 are relabeled based on the agent w/ the highest confidence
### if agent 1 (XGBoost) predicts w/ higher confidence, use its label
### otherwise, agent 2's (SVM) label is used
# 3. combine training data:
## a new training dataset is created for each agent by combining:
### train1 + train3
### train2 + train3
# 4. both agents are retrained w/ their respective combined datasets


# 1. prediction phase
# predict labels + posterior probabilities for train3 using both agents
train3_features = vectorizer.transform(train3_unlabeled['words']).toarray()
train3_features = scaler.transform(train3_features)
agent1_probs = agent_1.predict_proba(train3_features)
agent2_probs = agent_2.predict_proba(train3_features)

# get predictions + confidence scores
agent1_preds = agent_1.predict(train3_features)
agent2_preds = agent_2.predict(train3_features)
agent1_confidences = np.max(agent1_probs, axis=1)
agent2_confidences = np.max(agent2_probs, axis=1)

# 2. relabeling phase
new_labels = []
for i in range(len(train3_unlabeled)):
    if agent1_confidences[i] > agent2_confidences[i]:
        new_labels.append(agent1_preds[i])  # use agent 1's prediction
    else:
        new_labels.append(agent2_preds[i])  # use agent 2's prediction

# add the new labels to train3
train3_relabelled = train3_unlabeled.copy()
train3_relabelled['category'] = label_encoder.inverse_transform(new_labels)

print("Relabeling of Train3 is complete.")

# 3. combine training data
# train1 + train3
train1_combined = pd.concat([train1, train3_relabelled], axis=0)
# train2 + train3
train2_combined = pd.concat([train2, train3_relabelled], axis=0)

print(f"Train1 Combined: {train1_combined.shape}")
print(f"Train2 Combined: {train2_combined.shape}")

# 4. retrain agents
# extract features + labels for combined datasets
X_train1_combined = vectorizer.transform(train1_combined['words']).toarray()
y_train1_combined = label_encoder.transform(train1_combined['category'])

X_train2_combined = vectorizer.transform(train2_combined['words']).toarray()
y_train2_combined = label_encoder.transform(train2_combined['category'])

# scale the combined training features
X_train1_combined = scaler.fit_transform(X_train1_combined)
X_train2_combined = scaler.transform(X_train2_combined)

# retrain agent 1 (XGBoost)
agent_1.fit(X_train1_combined, y_train1_combined)

# retrain agent 2 (SVM)
agent_2.fit(X_train2_combined, y_train2_combined)

# evaluate retrained Agents
y_pred_agent1_retrained = agent_1.predict(X_test)
accuracy_agent1_retrained = accuracy_score(y_test, y_pred_agent1_retrained) * 100
print("Retrained Agent 1 (XGBoost) Evaluation on Test Dataset:")
print(f"Accuracy: {accuracy_agent1_retrained:.2f}%")
print(classification_report(y_test, y_pred_agent1_retrained, target_names=label_encoder.classes_))

y_pred_agent2_retrained = agent_2.predict(X_test)
accuracy_agent2_retrained = accuracy_score(y_test, y_pred_agent2_retrained) * 100
print("Retrained Agent 2 (SVM) Evaluation on Test Dataset:")
print(f"Accuracy: {accuracy_agent2_retrained:.2f}%")
print(classification_report(y_test, y_pred_agent2_retrained, target_names=label_encoder.classes_))



# ---------- STEP 4: VISUALIZATION/GRAPHS ----------
# create graphs to clearly see results!
# confusion matrix visualized as heatmap
# save classification reports to text files


# before mutual learning:
# agent 1 (XGBoost)
agent1_matrix_before = confusion_matrix(y_test, y_pred_agent1)
# agent 2 (SVM)
agent2_matrix_before = confusion_matrix(y_test, y_pred_agent2)

# after mutual learning:
# agent 1 (XGBoost)
agent1_matrix_after = confusion_matrix(y_test, y_pred_agent1_retrained)
# agent 2 (SVM)
agent2_matrix_after = confusion_matrix(y_test, y_pred_agent2_retrained)

# plot each matrix as heatmaps
# created this as a function to make it easier
def plot_confusion_matrix(cm, agent, phase, labels):
    # cm: confusion matrix
    # agent: name of agent (XGBoost or SVM)
    # phase: phase of mutual learning (before or after)
    # labels: list of category labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f'{agent} - {phase} Mutual Learning\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# define labels (category names)
labels = label_encoder.classes_

# plot confusion matrices
# before mutual learning:
plot_confusion_matrix(agent1_matrix_before, 'Agent 1 (XGBoost)', 'Before', labels)
plot_confusion_matrix(agent2_matrix_before, 'Agent 2 (SVM)', 'Before', labels)

plot_confusion_matrix(agent1_matrix_after, 'Agent 1 (XGBoost)', 'After', labels)
plot_confusion_matrix(agent2_matrix_after, 'Agent 2 (SVM)', 'After', labels)

# function to save classification reports results to a file
def save_results_to_file(agent, accuracy, classification_report_str, file_path):
    with open(file_path, 'a') as file:
        file.write(f"### {agent} Evaluation ###\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write(classification_report_str)
        file.write("\n" + "-"*50 + "\n")

# file path for saving the results
file_path = 'evaluation_results.txt'

# before mutual learning (initial training results)
# agent 1 (XGBoost) results
classification_report_agent1 = classification_report(y_test, y_pred_agent1, target_names=label_encoder.classes_)
# save results to file
save_results_to_file("Agent 1 (XGBoost)", accuracy_agent1, classification_report_agent1, file_path)

# agent 2 (SVM) results
classification_report_agent2 = classification_report(y_test, y_pred_agent2, target_names=label_encoder.classes_)
# save results to file
save_results_to_file("Agent 2 (SVM)", accuracy_agent2, classification_report_agent2, file_path)


# after mutual learning (retrained agents)
# agent 1 (XGBoost) retrained results
classification_report_agent1_retrained = classification_report(y_test, y_pred_agent1_retrained, target_names=label_encoder.classes_)
# save results to file
save_results_to_file("Retrained Agent 1 (XGBoost)", accuracy_agent1_retrained, classification_report_agent1_retrained, file_path)

# agent 2 (SVM) retrained results
classification_report_agent2_retrained = classification_report(y_test, y_pred_agent2_retrained, target_names=label_encoder.classes_)
# save results to file
save_results_to_file("Retrained Agent 2 (SVM)", accuracy_agent2_retrained, classification_report_agent2_retrained, file_path)

# Print message indicating results are saved
print(f"Evaluation results have been saved to {file_path}")