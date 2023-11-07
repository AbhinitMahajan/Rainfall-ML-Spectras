# Rain Prediction using Machine Learning

## Overview

This Jupyter notebook includes the implementation of machine learning models to predict rainfall based on various sensor data. The data is loaded from SQLite databases, preprocessed, and then used to train different neural network architectures. Additionally, feature importance is assessed using a Random Forest classifier and permutation importance. Overall, you will get a high-level overview of utilizing and comparing spectral(absorbance & reflectance) information acquired from the sewage plants of Dübendorf for predicting rainfall. 

## Data Loading

- `mvx_data`, `scan_data`, and `ise_data` are variables that hold the data loaded from corresponding SQLite databases.
- The data includes measurements from multiple sensors in different spectrums such as MVX, SCAN, and ISE.
- The data is loaded using a custom `SQLDataReader` class, which allows for querying within specific time ranges.

## Data Preprocessing

### Functions:

- `load_and_preprocess_rain_data`: Loads rain data from a CSV file, drops unnecessary columns, converts timestamps, and selects relevant columns- flow, level, rain 
- `drop_std_columns`: Removes any columns related to standard deviation as they are not needed for the analysis.
- `merge_with_rain_data`: Merges sensor(ICE, SCAN & MVX) data with rain data on timestamps and filters out entries with unspecified rain data.This function gets utilized inside the following function
- `preprocess_and_merge_data`: Orchestrates the loading, preprocessing, and merging of MVX, SCAN, and ISE data with rain data.
- `encode_and_scale_features`: Encodes the categorical 'rain' column using a label encoder and scales the remaining features to a range between 0 and 1 using min-max scaling.
#### Some takeaways:
- Two dataframes produced: mergedRS_df and mergedAS_df carrying spectral features from MVX & SCAN separately with some common features - flow, level, rain, ph, temp, nh4
- Standard deviation columns are dropped from the MVX data.
- Unwanted columns like 'nh4raw' and 'ph_raw' are dropped from the ISE data.
- The remaining data is merged with preprocessed rain data.
- User specifies which common columns or features need to be dropped in the mergedRS_df & mergedAS_df based on a list of columns to include as input 

## Neural Network Models

Five neural network architectures (`model1_architecture` to `model5_architecture`) are defined to handle different subsets of features and to test which configuration yields the best prediction results.

### Model Training and Evaluation

- `compile_and_train`: Compiles and trains the neural network model using binary cross-entropy loss, Adam optimizer, and includes precision and recall as metrics.
- `prepare_and_train`: Prepares the dataset by splitting it into training and test sets, selects an architecture, and proceeds with the training and evaluation process.
- Evaluation metrics such as accuracy, precision, recall, and F1-score are printed after training.

## Feature Importance Analysis

- A Random Forest classifier is trained on all the features/data to estimate feature importances based on it 
- Permutation importance for the neural architectures is computed using a custom scoring function based on the F1-score(combination of precision & recall)
- A combined plot shows both Random Forest feature importances and permutation importance for visual comparison in order to understand trends in seeing which spectral/ise features are important towards predicting rainfall

## Reflectancs vs Absorbancs using ML
here we make a comparison to see which spectral information(Absorbance or Reflectance) produces better prediction of rainfall. Before proceeding to this section, first extract mmergedAS_df and mergedRS_df with just the spectral features from the datapreprocessing section. Based on user input- we feed a dataframe either containing reflectance spectra or absorbance spectra to the model training
- `rename_columns`: A utility function to rename columns with a specified prefix based on which spectral information is being used, facilitating the distinction between different feature sets.
- `prepare_data`: Splits the dataset into training and test sets.
- `train_random_forest`: A function that applies grid search cross-validation to find the best hyperparameters for a RandomForestClassifier.
- `evaluate_model`: Evaluates the best model obtained from the grid search on the test set using precision and recall as the metrics.


## Execution Example

The notebook includes an example usage where specific architectures are selected, epochs and batch size are defined, and the model training is initiated. Following the training, the notebook presents feature importance analysis and concludes with hyperparameter tuning of a RandomForestClassifier. 

---

End-users of this notebook should replace file paths and drop columns as per their dataset's needs. Adjustments in the model architecture, epochs, batch size, and threshold for classification can be made to tailor the model's performance to new data or different use cases.
