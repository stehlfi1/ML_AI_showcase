# ML_AI_showcase

Projects are diced as some files were too big for github (full signal analysis project with frontend or just big csv or image files)

## Project: Airlines Delay Classification

This project focuses on predicting airline delays using a dataset obtained from Kaggle. The primary goal is to achieve the best possible prediction accuracy for flight delays.

### Dataset
- Dataset Source: [Kaggle Airlines Dataset](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay?datasetId=2285093&sortBy=voteCount)
- Description: The dataset contains information about airline flights, including features such as departure and arrival airports, flight details, and whether the flight experienced a delay.
- Target Variable: "Delay," a binary indicator (0 for no delay, 1 for delay).

### Project Highlights

#### Data Preprocessing
- Imported and explored the dataset, including data statistics and correlation analysis.
- One-hot encoded the "Airline" feature and applied label encoding to categorical features "AirportFrom" and "AirportTo."
- Split the data into training and testing sets for model evaluation.

#### Model Selection and Evaluation
- Conducted model evaluation using various classifiers, including Logistic Regression, K-Nearest Neighbors, Random Forest, Decision Tree, and CatBoost.
- Employed k-fold cross-validation to assess model performance.

#### Neural Network Experimentation
- Trained neural network models to explore alternative approaches.
- Tuned hyperparameters and experimented with different network architectures.
- Visualized training progress and evaluated model performance.




## Project: Australian Weather Prediction

This project is aimed at predicting weather conditions in Australia based on historical weather data. The primary goal is to develop accurate weather prediction models using various machine learning algorithms.

### Dataset
- Dataset Source: The dataset used in this project contains historical weather data for various locations in Australia. It includes features such as temperature, rainfall, humidity, wind speed, and more.
- Target Variable: "RainTomorrow," a binary indicator (0 for no rain tomorrow, 1 for rain tomorrow).

### Project Highlights

#### Data Preprocessing
- Imported the dataset and performed initial data exploration.
- Removed outliers from numerical features using the IQR method.
- Handled missing values using the K-nearest neighbors (KNN) imputer.
- Encoded categorical features, including location and wind direction, using one-hot encoding and ordinal encoding.

#### Model Selection and Evaluation
- Conducted model evaluation using various classifiers, including Logistic Regression, Random Forest, CatBoost, and more.
- Utilized k-fold cross-validation for robust model performance assessment.
- Achieved competitive accuracy scores with different classifiers.

### Model Ensemble
- Experimented with model ensemble techniques, including a Voting Classifier.
- Combined multiple classifiers to improve prediction accuracy.


## algos and agents
Older projects
- reinforcement learning
- markov decision proccess
- implementation of knn algo and bayes algo
- astar
## Project: Natural Images Classification

This project focuses on the classification of natural images into various categories using machine learning techniques. The primary objective is to create a model that can accurately distinguish between different types of natural images.

### Dataset
- Dataset Source: The dataset used in this project consists of natural images categorized into various classes, such as animals, flowers, and more. The dataset is sourced from Kaggle.
- Image Preprocessing: Images were resized to a common size (32x32 pixels) and standardized by scaling their pixel values to the range [0, 1].

### Project Highlights

#### Data Loading and Preprocessing
- Imported the dataset, which included images from different categories.
- Resized and standardized the images for consistency.
- Encoded the labels using one-hot encoding.

#### Model Building
- Constructed a convolutional neural network (CNN) model for image classification.
- Utilized multiple layers, including convolutional layers, max-pooling layers, dropout layers, and fully connected layers.
- Compiled the model using categorical cross-entropy loss and the Adam optimizer.

#### Model Training and Evaluation
- Split the dataset into training and testing sets.
- Trained the CNN model on the training data for 80 epochs.
- Evaluated the model's performance on the testing data.
- Visualized the training process, including accuracy and loss over epochs.

#### Model Evaluation Metrics
- Calculated accuracy, precision, recall, and F1-score to assess the model's performance.
- Generated a confusion matrix to visualize the model's predictions.

### Results
- The CNN model achieved competitive accuracy and demonstrated its ability to classify natural images into their respective categories.
- Classification reports and confusion matrices provide detailed insights into the model's performance for each category.

Feel free to explore the project code and experiment with different hyperparameters and architectures for image classification. Contributions and improvements are welcome!

## Signal Analysis for Artefact Detection
22 hours of medical signals with annotation of artefacts. Tested various aproaches before settling on knn classification. Aproach regarded as unique as it differes from classical aproaches (iso forrests, normal analysys (fourier), LSTM (all tried all failed))
Deployment of model for use.

This project involves signal analysis and machine learning techniques to detect artefacts in signal data. The project includes several key steps and components:

### Data Preprocessing
The project starts by loading and preparing the data. It reads XML files containing signal information and creates a DataFrame for further analysis.
Timestamps are generated and structured for each data point in the DataFrame.
### Data Integration
The project integrates data from multiple sources, including XML files and HDF5 files, to create a comprehensive dataset for analysis.
### Classification with Machine Learning
Several machine learning algorithms are applied to classify artefacts in the signal data, including Logistic Regression, K-Nearest Neighbors, Random Forest, Support Vector Machine, CatBoost, and XGBoost.
Pipelines with different preprocessing techniques, such as Min-Max scaling, are used to evaluate the performance of these classifiers.
### Evaluation Metrics
The project evaluates the classifiers using various metrics, including accuracy, F1-score, and classification reports.
F1-score is utilized to assess the models' precision and recall, providing a balanced measure of their performance.
