
# ## Importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import warnings
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pickle


warnings.filterwarnings('ignore')

# ## Dataset Exploration & Visualisation of different audio wafeforms

# The Genres present in the dataset

print("The genres present in the dataset are:")
print(os.listdir("Data/genres_original"))

# %% [markdown]
# Understanding the audion files

# %%
y, sr = librosa.load("Data/genres_original/blues/blues.00000.wav")

print("Sound Array :", y)
print("Sample Rate (KHz) =", sr)

# %% [markdown]
# Trimming the Silence sequences in the audio files

# %%
y, _ = librosa.effects.trim(y)
print("After trimming the silence sequences the sound array is:")
print(y)

# Observe no preceeding or succeding silence

# %% [markdown]
# 2D Representation of audio using Waveform

# %%
plt.figure(figsize = (16,6))
librosa.display.waveshow(y = y, sr = sr, color = "#FF00AB")
plt.title("Waveform in Blues 0", fontsize = 24)

# %% [markdown]
# Decomposing the waveform based on the frequencies using Short time Fouries Transform

# %%
n_fft = 2048
hop_length = 512

D = np.abs(librosa.stft(y, n_fft = n_fft, hop_length = hop_length))

plt.plot(D)
plt.show()

# %% [markdown]
# Creating the Log Frequency Spectrogram from the transformed Signal

# %%
# This also scales to the decibel system, as that is also log based.

Deci = librosa.amplitude_to_db(D, ref = np.max)
librosa.display.specshow(Deci, sr = sr, hop_length=hop_length, x_axis='time',y_axis='log',cmap = 'cool')

plt.colorbar()
plt.title("Log Frequency Spectrogram")

# %% [markdown]
# Plotting Mel Spectrogram from the data

# %%
Mel_S = librosa.feature.melspectrogram(y =y, sr= sr)
Deci_S = librosa.amplitude_to_db(Mel_S, ref = np.max)
librosa.display.specshow(Deci_S, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',cmap = 'cool')
plt.colorbar()
plt.title("Mel Spectrogram", fontsize = 12)

# %% [markdown]
# Plotting Chroma Spectrogram which represents the energy distribution across pitch classes

# %%
# Lets compare 3 genres, Rock, Pop and Classical

rock, r_sr = librosa.load("Data/genres_original/rock/rock.00000.wav")
classical, c_sr = librosa.load("Data/genres_original/classical/classical.00000.wav")
pop, p_sr = librosa.load("Data/genres_original/pop/pop.00000.wav")

# Compute chroma spectrograms for each genre
chroma_rock = librosa.feature.chroma_cqt(y=rock, sr=r_sr)
chroma_cls = librosa.feature.chroma_cqt(y=classical, sr=c_sr)
chroma_pop = librosa.feature.chroma_cqt(y=pop, sr=p_sr)

# Plot the chroma spectrograms for each genre
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
librosa.display.specshow(chroma_rock, sr=r_sr, x_axis='time', y_axis='chroma')
plt.title('Chroma Spectrogram - Rock')
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 2)
librosa.display.specshow(chroma_cls, sr=c_sr, x_axis='time', y_axis='chroma')
plt.title('Chroma Spectrogram - Classical')
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 3)
librosa.display.specshow(chroma_pop, sr=p_sr, x_axis='time', y_axis='chroma')
plt.title('Chroma Spectrogram - Pop')
plt.colorbar()
plt.tight_layout()

plt.show()

# %% [markdown]
# As can be seen through the chroma spectogram, the Rock and Classical Music seem to have the melodies consisting to a lot of notes at the same time, while the pop music(for general audience) consist very few notes at the same instant.
# Even if multiple notes are present at same time in Pop, they are usually harmonies i.e. 3rd or 5th of the root note.

# %% [markdown]
# Plotting Tempograms for a few genres

# %%
tempo_rock= librosa.beat.tempo(y=rock, sr=r_sr)
tempogram_rock = librosa.feature.tempogram(y=rock, sr=r_sr, hop_length=512, win_length=384, window=np.hanning)

tempo_cls = librosa.beat.tempo(y=classical, sr=c_sr)
tempogram_cls = librosa.feature.tempogram(y=classical, sr=c_sr, hop_length=512, win_length=384, window=np.hanning)

tempo_pop = librosa.beat.tempo(y=pop, sr=p_sr)
tempogram_pop = librosa.feature.tempogram(y=pop, sr=p_sr, hop_length=512, win_length=384, window=np.hanning)

# Plot the tempograms for each genre
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
librosa.display.specshow(tempogram_rock, sr=r_sr, hop_length=512, x_axis='time', y_axis='tempo')
plt.title('Tempogram - Rock (Tempo: {:.2f} BPM)'.format(tempo_rock[0]))
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 2)
librosa.display.specshow(tempogram_cls, sr=c_sr, hop_length=512, x_axis='time', y_axis='tempo')
plt.title('Tempogram - Classical (Tempo: {:.2f} BPM)'.format(tempo_cls[0]))
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 3)
librosa.display.specshow(tempogram_pop, sr=p_sr, hop_length=512, x_axis='time', y_axis='tempo')
plt.title('Tempogram - Pop (Tempo: {:.2f} BPM)'.format(tempo_pop[0]))
plt.colorbar()
plt.tight_layout()

plt.show()

# %% [markdown]
# Not much can be said able the genres with just looking at the tempograms,
# so let's overlay the tempogram with the CQT spectogram, which provides the pitch over time for the audio file.

# %% [markdown]
# Plotting CQT Spectrograms for a few genres

# %%
cqt_rock = np.abs(librosa.cqt(rock, sr=r_sr, hop_length=512))
cqt_cls = np.abs(librosa.cqt(classical, sr=c_sr, hop_length=512))
cqt_pop = np.abs(librosa.cqt(pop, sr=p_sr, hop_length=512))

# Plot the CQT spectrograms with tempograms overlaid for each genre
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(cqt_rock, ref=np.max), sr=r_sr, hop_length=512, x_axis='time', y_axis='cqt_note')
librosa.display.specshow(tempogram_rock, sr=r_sr, hop_length=512, x_axis='time', y_axis='tempo', cmap='magma', alpha=0.6)
plt.title('Rock CQT Spectrogram with Tempogram (Tempo: {:.2f} BPM)'.format(tempo_rock[0]))
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(cqt_cls, ref=np.max), sr=c_sr, hop_length=512, x_axis='time', y_axis='cqt_note')
librosa.display.specshow(tempogram_cls, sr=c_sr, hop_length=512, x_axis='time', y_axis='tempo', cmap='magma', alpha=0.6)
plt.title('Classical CQT Spectrogram with Tempogram (Tempo: {:.2f} BPM)'.format(tempo_cls[0]))
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(cqt_pop, ref=np.max), sr=p_sr, hop_length=512, x_axis='time', y_axis='cqt_note')
librosa.display.specshow(tempogram_pop, sr=p_sr, hop_length=512, x_axis='time', y_axis='tempo', cmap='magma', alpha=0.6)
plt.title('Pop CQT Spectrogram with Tempogram (Tempo: {:.2f} BPM)'.format(tempo_pop[0]))
plt.colorbar()
plt.tight_layout()

plt.show()


# %% [markdown]
# Harmonics & Percussive for the audio files

# %%
y_harm, y_perc = librosa.effects.hpss(y)
plt.plot(y_harm, color = '#DD22AA')
plt.plot(y_perc, color = '#11DD44')
plt.title("Harmonics & Percussive")
plt.show()

# %%
data = pd.read_csv("Data/features_3_sec.csv")
print(data.shape)

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# Having a look at the data & the information about its features

# %%
data.head()

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# #### CorrelationHeatmap

# %%
# Heat map for the mean variables
mean_cols = [col for col in data.columns if 'mean' in col]

corr = data[mean_cols].corr()
f, ax = plt.subplots(figsize=(16, 11))

sns.heatmap(corr, cmap = 'Blues')
plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)

plt.show()

# %%
# Heat map for Variance variables
var_cols = [col for col in data.columns if 'var' in col]

corr = data[var_cols].corr()
f, ax = plt.subplots(figsize=(16, 11))

sns.heatmap(corr, cmap = 'Blues')
plt.title('Correlation Heatmap (for the VARIANCE variables)', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)

plt.show()

# %% [markdown]
# ## Preprocessing

# %%
data.drop(['filename','length'], axis = 1, inplace = True)

# %%
data.head()

# %%
X = data.iloc[:,:-1]

# %%
Y = data.iloc[:,-1]

# %%
Y

# %% [markdown]
# #### Scaling the features of data

# %%
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X)

# %%
columns = X.columns

# %%
X_scaled = pd.DataFrame(scaled_data, columns = columns)

# %%
X_scaled.head()

# %% [markdown]
# #### Implementing Principal Component Analysis

# %% [markdown]
# Checking the Variance explained by different number of components

# %%
for i in range(1,57):
    pca_dummy = PCA(n_components = i)

    dummy_trans = pca_dummy.fit_transform(X_scaled)

    print("For", i,"number of components Explained Variance Ratio =", pca_dummy.explained_variance_ratio_.cumsum()[i-1])

# %% [markdown]
# Since, 90% of the variance can be explained by taking 24 components so, the components are reduced to 24

# %%
# 90% of variance can be explained using 24 components
pca = PCA(n_components = 24)
components = pca.fit_transform(X_scaled)

X_pca = pd.DataFrame(components)

# %% [markdown]
# Splitting the reduced data in train & test part

# %%
X_train, X_test, y_train, y_test = train_test_split(X_pca,Y, test_size=0.2, random_state=30)

# %%
X_train

# %% [markdown]
# ## Implementing different models

# %% [markdown]
# #### Implementing K Nearest Neighbors

# %% [markdown]
# Grid Search for finding best parameters for K Nearest Neighbors

# %%
# Setting up the parameter grid for KNN
param_found_knn = 1

knn_params = {
    'n_neighbors': range(1, 21),  # Considering a range that isn't too broad
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']  # Adding different distance metrics
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', verbose=1)

if not param_found_knn:
    knn_grid.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters for KNN:", knn_grid.best_params_)
    print("Best cross-validation score for KNN:", knn_grid.best_score_)

else:
    knn_grid.best_params_ = {'metric': 'manhattan', 'n_neighbors': 4, 'weights': 'distance'}


# %% [markdown]
# Best parameters for KNN: {'metric': 'manhattan', 'n_neighbors': 4, 'weights': 'distance'}

# %%
# Training KNN with the best parameters
knn_best = KNeighborsClassifier(**knn_grid.best_params_)
knn_best.fit(X_train, y_train)
y_pred_knn_test = knn_best.predict(X_test)
y_pred_knn_train = knn_best.predict(X_train)

# Evaluation
knn_accuracy_test = accuracy_score(y_test, y_pred_knn_test)
knn_accuracy_train = accuracy_score(y_train, y_pred_knn_train)
knn_report = classification_report(y_test, y_pred_knn_test, output_dict=True)
print("Accuracy using KNN =", knn_accuracy_test)

# %% [markdown]
# Classfication Report for K Nearest Neighbors

# %%
# Transform classification report into DataFrame
knn_report_df = pd.DataFrame(knn_report).transpose()

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(data=knn_report_df.iloc[:-1, :].drop(columns=['support']), annot=True, cmap='Blues', fmt='.2f')
plt.title('KNN Classification Report')
plt.show()

# %% [markdown]
# Confusion Matrix for K Nearest Neighbors

# %%
# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_knn_test), annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.show()

# %% [markdown]
# #### Implementing Decision Trees

# %% [markdown]
# Grid Search for finding best parameters for Decision Trees

# %%
# Setting up the parameter grid for Decision Trees

param_found_dt = 1
dt_params = {
    'max_depth': range(5, 10),  # Max depth to prevent overfitting
    'min_samples_split': range(2, 10, 2),  # Moderate range for minimum samples split
    'min_samples_leaf': range(1, 5)  # Minimum samples per leaf to ensure generalization
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy', verbose=1)

if not param_found_dt:
    dt_grid.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters for Decision Tree:", dt_grid.best_params_)
    print("Best cross-validation score for Decision Tree:", dt_grid.best_score_)

else:
    dt_grid.best_params_ = {'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 4}

# %% [markdown]
# Best parameters for Decision Tree: {'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 4}

# %%
# Training Decision Tree with the best parameters
dt_best = DecisionTreeClassifier(**dt_grid.best_params_, random_state=42)
dt_best.fit(X_train, y_train)
y_pred_dt_test = dt_best.predict(X_test)
y_pred_dt_train = dt_best.predict(X_train)

# Evaluation
dt_accuracy_test = accuracy_score(y_test, y_pred_dt_test)
dt_accuracy_train = accuracy_score(y_train, y_pred_dt_train)
dt_report = classification_report(y_test, y_pred_dt_test, output_dict=True)
print("Accuracy using Decision Tree =", dt_accuracy_test)

# %% [markdown]
# Classification Report for Decision Trees

# %%
# Transform classification report into DataFrame
dt_report_df = pd.DataFrame(dt_report).transpose()

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(data=dt_report_df.iloc[:-1, :].drop(columns=['support']), annot=True, cmap='Blues', fmt='.2f')
plt.title('Decision Tree Classification Report')
plt.show()


# %% [markdown]
# Confusion Matrix for Decision Trees

# %%
# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_dt_test), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# %% [markdown]
# #### Implementing Support Vector Machines

# %% [markdown]
# Grid Search for finding best parameters for Support Vector Machines

# %%
# Setting up the parameter grid for SVM

param_found_svm = 1
svm_params = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['rbf', 'poly', 'sigmoid']  # Different types of kernels
}
svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring='accuracy', verbose=1)
if not param_found_svm:

    svm_grid.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters for SVM:", svm_grid.best_params_)
    print("Best cross-validation score for SVM:", svm_grid.best_score_)

else:
    svm_grid.best_params_ = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}


# %% [markdown]
# Best parameters for SVM: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

# %%
# Training SVM with the best parameters
svm_best = SVC(**svm_grid.best_params_, random_state=42)
svm_best.fit(X_train, y_train)
y_pred_svm_test = svm_best.predict(X_test)
y_pred_svm_train = svm_best.predict(X_train)

# Evaluation
svm_accuracy_test = accuracy_score(y_test, y_pred_svm_test)
svm_accuracy_train = accuracy_score(y_train, y_pred_svm_train)
svm_report = classification_report(y_test, y_pred_svm_test, output_dict=True)

print("Accuracy using SVM =", svm_accuracy_test)

# %% [markdown]
# Classification Report for Support Vector Machines

# %%
# Transform classification report into DataFrame
svm_report_df = pd.DataFrame(svm_report).transpose()

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(data=svm_report_df.iloc[:-1, :].drop(columns=['support']), annot=True, cmap='Blues', fmt='.2f')
plt.title('SVM Classification Report')
plt.show()


# %% [markdown]
# Confusion Matrix for Support Vector Machines

# %%
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_svm_test), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()

# %% [markdown]
# #### Implementing Adaboost

# %% [markdown]
# Grid Search for finding best parameters for Adaboost

# %%
param_found_ada = 1
adaboost_params = {
    'n_estimators': [50, 100, 150, 200], # Number of models to iteratively train
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1] # Weight applied to each classifier at each boosting iteration
}
adaboost_grid = GridSearchCV(AdaBoostClassifier(random_state=42), adaboost_params, cv=5, scoring='accuracy', verbose=1)
if not param_found_ada:
# Setting up the parameter grid for Adaboost

    adaboost_grid.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters for Adaboost:", adaboost_grid.best_params_)
    print("Best cross-validation score for Adaboost:", adaboost_grid.best_score_)
else:
    adaboost_grid.best_params_ = {'learning_rate': 0.5, 'n_estimators': 50}

# %% [markdown]
# Best parameters for Adaboost: {'learning_rate': 0.5, 'n_estimators': 50}

# %%
# Training Adaboost with the best parameters
adaboost_best = AdaBoostClassifier(**adaboost_grid.best_params_, random_state=42)
adaboost_best.fit(X_train, y_train)
y_pred_adaboost_test = adaboost_best.predict(X_test)
y_pred_adaboost_train = adaboost_best.predict(X_train)

# Evaluation
adaboost_accuracy_test = accuracy_score(y_test, y_pred_adaboost_test)
adaboost_accuracy_train = accuracy_score(y_train, y_pred_adaboost_train)
adaboost_report = classification_report(y_test, y_pred_adaboost_test, output_dict=True)

print("Accuracy using Adaboost =", adaboost_accuracy_test)

# %% [markdown]
# Classification Report for Adaboost

# %%
# Transform classification report into DataFrame
adaboost_report_df = pd.DataFrame(adaboost_report).transpose()

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(data=adaboost_report_df.iloc[:-1, :].drop(columns=['support']), annot=True, cmap='Blues', fmt='.2f')
plt.title('Adaboost Classification Report')
plt.show()

# %% [markdown]
# Confusion Matrix for Adaboost

# %%
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_adaboost_test), annot=True, fmt='d', cmap='Blues')
plt.title('Adaboost Confusion Matrix')
plt.show()

# %% [markdown]
# #### Implementing Logistic Regression

# %% [markdown]
# Grid Search for finding best parameters for Logistic Regression

# %%
# Setting up the parameter grid for Logistic Regression
param_found_lr = 1

lr_params = {
    'C': [0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Algorithm to use in the optimization problem
    'max_iter': [100, 200, 300]  # Maximum number of iterations taken for the solvers to converge
}
lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5)

if not param_found_ada:
    lr_grid.fit(X_train, y_train)

    # Best parameters and best score
    print("Best parameters for Logistic Regression:", lr_grid.best_params_)
    print("Best cross-validation score for Logistic Regression:", lr_grid.best_score_)

else:
    lr_grid.best_params_= {'C': 100, 'max_iter': 100, 'solver': 'lbfgs'}


# %% [markdown]
# Best parameters for Logistic Regression: {'C': 100, 'max_iter': 100, 'solver': 'lbfgs'}

# %%
# Training Logistic Regression with the best parameters
lr_best = LogisticRegression(**lr_grid.best_params_, random_state=42)
lr_best.fit(X_train, y_train)
y_pred_lr_test = lr_best.predict(X_test)
y_pred_lr_train = lr_best.predict(X_train)

# Evaluation
lr_accuracy_test = accuracy_score(y_test, y_pred_lr_test)
lr_accuracy_train = accuracy_score(y_train, y_pred_lr_train)
lr_report = classification_report(y_test, y_pred_lr_test, output_dict=True)

print("Accuracy using Logistic Regression =", lr_accuracy_test)

# %% [markdown]
# Classification Report for Logistic Regression

# %%
# Transform classification report into DataFrame
lr_report_df = pd.DataFrame(lr_report).transpose()

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(data=lr_report_df.iloc[:-1, :].drop(columns=['support']), annot=True, cmap='Blues', fmt='.2f')
plt.title('Logistic Regression Classification Report')
plt.show()

# %% [markdown]
# Confusion Matrix for Logistic Regression

# %%
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr_test), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# %% [markdown]
# #### Implementing Artificial Neural Networks

# %% [markdown]
# Encoding the target values for application of ANN

# %%
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# %% [markdown]
# Applying the model with the hidden layers with different number of neurons

# %%
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %% [markdown]
# Saving the Weights & Bias of the trained model in genre_classifier.keras file for ease of computation

# %%
try :
    class History():
        def __init__(self, history):
            self.history = history
    model = (load_model('genre_classifier.keras'))
    with open('model_history.pkl', 'rb') as file:
        loaded_history = pickle.load(file)
    history = History(loaded_history)
    
        
except:
    history = model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), verbose=0)
    with open('model_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

# %%
model.save('genre_classifier.keras')

# %% [markdown]
# Plotting the Loss Curve with epochs

# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# Plotting the Accuracy Curve with epochs

# %%
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% [markdown]
# Function to predict the Genre of audio

# %%
def predictANN(data):
    return encoder.classes_[np.argmax(model.predict(data), axis=1)]

# %% [markdown]
# Computing the train & test accuracy

y_test_pred_ANN = np.argmax(model.predict(X_test), axis=1)
ann_accuracy_test = accuracy_score(y_test_pred_ANN, y_test)
print("Test accuracy using ANN =", ann_accuracy_test)

y_train_pred_ANN = np.argmax(model.predict(X_train), axis=1)
ann_accuracy_train = accuracy_score(y_train_pred_ANN, y_train)
print("Train accuracy using ANN =", ann_accuracy_train)

# Classification Report for Artificial Neural Networks

ANN_report = classification_report(y_test, y_test_pred_ANN, output_dict=True)

# Transform classification report into DataFrame
ANN_report_df = pd.DataFrame(ANN_report).transpose()

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(data=lr_report_df.iloc[:-1, :].drop(columns=['support']), annot=True, cmap='Blues', fmt='.2f')
plt.title('ANN Classification Report')
plt.show()

# Confusion Matrix for Artificial Neural Networks

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_test_pred_ANN), annot=True, fmt='d', cmap='Blues')
plt.title('ANN Confusion Matrix')
plt.show()

models = ['KNN', 'Decision Tree', 'SVM', 'Adaboost', 'Logistic Regression', 'ANN']

index = np.arange(len(models))
bar_width = 0.35

# Corresponding accuracies
test_accuracies = [knn_accuracy_test, dt_accuracy_test, svm_accuracy_test, adaboost_accuracy_test, lr_accuracy_test, ann_accuracy_test]
train_accuracies = [knn_accuracy_train, dt_accuracy_train, svm_accuracy_train, adaboost_accuracy_train, lr_accuracy_train, ann_accuracy_train]

# Creating the bar chart
plt.figure(figsize=(10, 6))

plt.bar(index, test_accuracies, bar_width, label='Test Accuracy', color="skyblue", edgecolor='black')
plt.bar(index + bar_width, train_accuracies, bar_width, label='Train Accuracy', color="mistyrose", edgecolor='black')

# Adding text labels
for i in range(len(index)):
    plt.text(i, test_accuracies[i] + 0.005, f'{test_accuracies[i]*100:.2f}%', ha='center', va='bottom', fontsize=10)
    plt.text(i + bar_width, train_accuracies[i] + 0.005, f'{train_accuracies[i]*100:.2f}%', ha='center', va='bottom', fontsize=10)

plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparison of Model Accuracies', fontsize=14)
plt.xticks(index + bar_width / 2, models, rotation=45, ha='right', fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()

path = "Data/genres_original/blues/blues.00000.wav"

def music_transform(path):

    if ".mp3" in path:
        y, sr = sf.read(path)

        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)

    y, sr = librosa.load(path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    mfccs_mean = np.mean(mfccs,axis = 1)
    mfccs_var = np.var(mfccs, axis = 1)
    rms = librosa.feature.rms(y=y)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo = librosa.beat.tempo(y=y, sr=sr)

    data_array = []
    features = [chromagram,rms,spectral_centroid,spectral_bandwidth,spectral_rolloff,zero_crossing_rate,y_harmonic,y_percussive]

    for i in features:
        data_array.append(np.mean(i))
        data_array.append(np.var(i))

    data_array.append(tempo)

    for i in range(20):
        data_array.append(mfccs_mean[i])
        data_array.append(mfccs_var[i])

    music_data = pd.DataFrame(np.array(data_array).reshape(1,57),columns = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
       'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'])
    scaled_data_music = scaler.transform(music_data)

    scaled_dataframe = pd.DataFrame(scaled_data_music, columns = music_data.columns)

    musicpca_components = pca.transform(scaled_dataframe)

    music_pca = pd.DataFrame(musicpca_components)
    return music_pca



# %% [markdown]
# Function which takes path of audio file and name of classifier to predict the Genre

# %%
def modelPrediction(path, classifier = "KNN"):

    music_data = music_transform(path)

    if classifier == "KNN":
        return knn_best.predict(music_data)
    elif classifier == "Decision Tree":
        return dt_best.predict(music_data)
    elif classifier == "SVM":
        return svm_best.predict(music_data)
    elif classifier == "Adaboost":
        return adaboost_best.predict(music_data)
    elif classifier == "Logistic Regression":
        return lr_best.predict(music_data)
    elif classifier == "ANN":
        return predictANN(music_data)
    else:
        return "Model Not Found"

# %%
print("According to KNN the audio is", modelPrediction(path,"KNN"))
print("According to Decision Tree the audio is", modelPrediction(path,"Decision Tree"))
print("According to SVM the audio is", modelPrediction(path,"SVM"))
print("According to Adaboost the audio is", modelPrediction(path,"Adaboost"))
print("According to Logistic Regression the audio is", modelPrediction(path,"Logistic Regression"))
print("According to ANN the audio is", modelPrediction(path,"ANN"))


