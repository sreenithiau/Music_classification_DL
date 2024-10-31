
# Music Genre Classification

This project implements machine learning algorithms to automatically classify music into different genres. It explores techniques for feature extraction from audio data and training models to identify genres based on these features.


## WebPage

Visit our project website for a detailed overview of the materials and a comprehensive understanding of the project goals and objectives.

## Dataset

For the purpose of classification, we have used the GTZAN dataset, a popular benchmark dataset for music genre classification tasks.

The GTZAN dataset consists of 1,000 audio recordings, each 30 seconds long, categorized into 10 distinct music genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. All audio files are in WAV format with a sampling rate of 22,050 Hz and mono channel.

This dataset provides a well-structured and diverse collection of music genres, making it suitable for training and evaluating machine learning models for music genre classification.

We performed the following tasks on our project:
- Feature Extraction with PCA (Principal Component Analysis): We are extracting relevant features from the audio data using Principal Component Analysis (PCA). PCA helps reduce dimensionality by identifying the most informative features in the data, which can improve model performance and training efficiency.

- Classification Model Evaluation: To establish a baseline and compare performance, we are training and evaluating several machine learning models on the extracted features. These models include K-Nearest Neighbors (KNN), Decision Trees, Support Vector Machines (SVM), Adaboost, Logistic Regression and Artificial Neural Networks. Evaluating their accuracy on a validation set will allow us to compare different approaches and find out which model performs the best.
## GitHub Repository

- Data Folder: The "data" folder contains the GTZAN dataset, including two CSV files with audio features for various genres. Additionally, it houses sub-folders containing audio files and their corresponding waveform images.

- music_classification.ipynb: This file serves as the core codebase for audio genre classification, implementing algorithms and processes for the classification task.

- genre_classifier.keras: This file stores the pre-trained weights and biases for the Artificial Neural Network (ANN) model, enabling faster execution of the main code file for audio genre classification.

- Mid-Progress Report.pdf: This file provides a detailed description of the dataset utilized and outlines the proposed methods for achieving music genre classification.

- model_history.pkl: It keeps a record of the Artificial Neural Network model's history, enabling the visualization of loss and accuracy curves over epochs.

- pca.sav: It retains the Principal Component Analysis (PCA) model trained on the GTZAN dataset for transforming any new audio file in web app.

- prediction_web_app.py: An interactive web application showcasing the top-performing model allows users to check the genre of any audio file they upload.

- scaler.sav: It retains the MinMaxScaler model trained on the GTZAN dataset for transforming any new audio file in web app.

- trained_model.sav: It preserves the top-performing model for predicting the genres of new audio files within the web application, ensuring accurate and efficient classification.

- report.pdf: It gives a detailed explanation of all the work done so far including the dataset explanation, implemented approaches, libraries and tech stack used, etc.

- index.html: This file comprises HTML code detailing a project overview webpage.

- style.css: This file consists of the css code written for styling of the project webpage.
## Tech Stack

- Python
- Scikit-Learn
- TensorFlow
- Librosa
- Streamlit
- HTML
- CSS
## Contributors
Sreenithi A
Shrivarshini K P 
Bakkyalakshmi
