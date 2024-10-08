# Stock Market Analysis and Prediction With Python :-📈 

**overview of ther Project :-**

# Problem Statement:

The goal of this report is to develop a model that can predict future stock prices using historical data. Accurate stock price predictions can help investors make better decisions, minimize risks, and maximize returns. The challenge lies in handling the complex, volatile nature of stock markets and extracting meaningful patterns from historical data.
The primary objective is to predict future stock prices based on historical stock data. The challenges include dealing with the volatile nature of the stock market, identifying relevant patterns, and building a model that can generalize well to new data. 

# Data Points:

To predict stock prices, we need the following data points:

- **Date:**  The specific trading day.

- **Open Price:**  The price of the stock at market open.

- **High Price:**  The highest price of the stock during the trading day.

- **Low Price:**  The lowest price of the stock during the trading day.

- **Close Price:**  The price of the stock at market close.

- **Volume:**  The number of shares traded during the day.

These data points are typically collected from financial websites, stock exchanges, or trading platforms.

# Approach:

The approach involves several key steps:

- **Data Collection:**  Gather historical stock price data for the desired stock(s).

- **Data Preprocessing:**  Clean the data to handle missing values and normalize it to ensure all features contribute equally to the model.

- **Feature Engineering:**  Create additional features that may help the model, such as moving averages, volatility, or momentum indicators.

- **Model Selection:**  Choose an appropriate machine learning model for prediction.

- **Model Training:**  Train the model using historical data.

- **Model Evaluation:**  Assess the model's performance using evaluation metrics.

- **Prediction:**  Use the trained model to predict future stock prices.

# Implementation:

Here in this project i have use the several python library so perform the project .The library are import pandas as pd, import math, import numpy as np .
here the import is used to make code available from one module top another .

**1 import pandas as pd :-** Pandas is usually imported under the pd alias. alias:

**2 import math:-** Python has also a built-in module called math , which extends the list of mathematical functions

**3 import Numpy as np:-** a library that provides a set of high level functions and features for performing data analysis and manipulation

Now, the next is 

 from IPython.display import display
 from sklearn import linear_model
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_squared_error
 from sklearn.model_selection import TimeSeriesSplit 
 from sklearn.preprocessing import MinMaxScaler

1 from IPython.display import display :- This line imports the display function from the IPython.display module, used to render rich output formats in Jupyter Notebook.

2 from sklearn import linear_model :- This imports the linear_model module from the sklearn package, which contains various linear regression models.

3 from sklearn.model_selection import train_test_split :- This imports the train_test_split function from the sklearn.model_selection module, used to split data into training and testing sets.

4 from sklearn.metrics import mean_squared_error :- This imports the mean_squared_error function from the sklearn.metrics module, used to compute the mean squared error between predicted and true values.

5 from sklearn.model_selection import TimeSeriesSplit :- This imports the TimeSeriesSplit class from the sklearn.model_selection module, used for splitting time series data into train and test sets without shuffling.

6 from sklearn.preprocessing import MinMaxScaler:- This imports the MinMaxScaler class from the sklearn.preprocessing module, used for scaling numerical features to a specified range (usually between 0 and 1).

The next part is :

**1 import preprocess_data as ppd** :- This imports the preprocess_data module and assigns it an alias ppd, likely containing functions for data preprocessing tasks.

**2 import visualize as vs :-** This imports the visualize module and assigns it an alias vs, likely containing functions for data visualization tasks.

**3 import stocks_data as sd :-** This imports the stocks_data module and assigns it an alias sd, likely containing functions or data related to stock market data manipulation or retrieval.

Then we have taken a .csv file here by showing   
df = pd.read_csv('google.csv')
df
here df means dataframe and pd.read means we are reading the data from the .csv file [google.csv] is a file name that we have imported, it is basically a excel file were all the data is stored in form of rows n columns 

Date	Open	High	Low	Close	Volume
0	30-Jun-17	943.99	945.00	929.61	929.68	2287662
1	29-Jun-17	951.35	951.66	929.60	937.82	3206674
2	28-Jun-17	950.66	963.24	936.16	961.01	2745568
3	27-Jun-17	961.60	967.22	947.09	948.09	2443602
4	26-Jun-17	990.00	993.99	970.33	972.09	1517912
...	...	...	...	...	...	...
3140	7-Jan-05	95.42	97.22	94.48	97.02	9666175
3141	6-Jan-05	97.72	98.05	93.95	94.37	10389803
3142	5-Jan-05	96.82	98.55	96.21	96.85	8239545
3143	4-Jan-05	100.77	101.57	96.84	97.35	13762396
3144	3-Jan-05	98.80	101.92	97.83	101.46	15860692

stocks = ppd.remove_data(df) :- It is used to call a function named remove_data from the preprocess_data module (imported as ppd). This function is applied to a DataFrame df and returns a modified DataFrame, which is then assigned to the variable stocks.

vs.plot_basic(stocks) :- It is used to call a function named plot_basic from the visualize module (imported as vs). This function is applied to the stocks DataFrame, which contains your preprocessed stock market data.

stocks = ppd.get_normalised_data(stocks)
print(stocks.head())

print("\n")
print("Open   --- mean :", np.mean(stocks['Open']),  "  \t Std: ", np.std(stocks['Open']),  "  \t Max: ", np.max(stocks['Open']),  "  \t Min: ", np.min(stocks['Open']))
print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ", np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
print("Volume --- mean :", np.mean(stocks['Volume']),"  \t Std: ", np.std(stocks['Volume']),"  \t Max: ", np.max(stocks['Volume']),"  \t Min: ", np.min(stocks['Volume']))
Here is explanation of each line .

# 1. Normalization:-
 stocks = ppd.get_normalised_data(stocks)

This line calls the get_normalised_data function from the preprocess_data module (imported as ppd). It normalizes the stocks DataFrame, adjusting the data to a common scale without distorting differences in the ranges of values.

# 2. Printing the Head of the DataFrame:
 print(stocks.head())

This line prints the first five rows of the stocks DataFrame. Printing the head of the DataFrame allows you to inspect the first few entries of your normalized data, ensuring that the normalization process was applied correctly.

# 3. Printing Statistical Summaries:
 print("\n")
 print("Open   --- mean :", np.mean(stocks['Open']),  "  \t Std: ", np.std(stocks['Open']),  "  \t Max: ", np.max(stocks['Open']),  "  \t Min: ", np.min(stocks['Open']))
 print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ", np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
 print("Volume --- mean :", np.mean(stocks['Volume']),"  \t Std: ", np.std(stocks['Volume']),"  \t Max: ", np.max(stocks['Volume']),"  \t Min: ", np.min(stocks['Volume']))

These lines print the mean, standard deviation, maximum, and minimum values for the 'Open', 'Close', and 'Volume' columns of the stocks DataFrame. These statistical summaries provide insights into the distribution and scale of the normalized data. Checking these statistics helps verify that the normalization process has adjusted the values as expected and ensures that the data is ready for further analysis or modeling.

null_value = stocks.isnull().sum():- These Line is used to check for and count any missing values (null values) in the stocks DataFrame.

vs.plot_basic(stocks):- The line calls the plot_basic function from the visualize module (imported as vs) and applies it to the stocks DataFrame.
                        By visualizing the data with vs.plot_basic(stocks), you ensure that you have a clear and comprehensive understanding of the stock data before moving on to more     
                        complex analyses or predictive modeling.

Next part is , 

from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

stocks = ppd.get_normalised_data(stocks)
print(stocks.head())
 Assuming `stocks` DataFrame is already defined and loaded with data
 Remove the 'Item' column from the data
stocks_data = stocks.drop(['Item'], axis=1)

The provided code is part of a workflow to preprocess stock market data, normalize it, and prepare it for modeling using a neural network

# 1. Importing Libraries and Modules:- 

from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

These lines import necessary modules and functions for building and training a neural network model using Keras, and for data preprocessing and evaluation using Scikit-Learn.

~ **Dense, Activation, Dropout, LSTM:** Layers used to build neural network models.

~ **Sequential:** Model type used in Keras for creating neural networks layer by layer.

~ **train_test_split:** Function to split data into training and testing sets.

~ **mean_squared_error:** Metric to evaluate model performance.

~ **StratifiedKFold:** Function for stratified k-fold cross-validation.

# 2. Normalize the Data:- 

stocks = ppd.get_normalised_data(stocks)
print(stocks.head())

~ To normalize the stocks DataFrame so that all features have a similar scale, which is crucial for the effective training of machine learning models. get_normalised_data is a function from the preprocess_data module that normalizes the stock data, ensuring that the features are on a similar scale.

# 3. Drop Unnecessary Column:- 

Assuming `stocks` DataFrame is already defined and loaded with data
Remove the 'Item' column from the data
stocks_data = stocks.drop(['Item'], axis=1)

~ To remove the 'Item' column from the stocks DataFrame, as it is not needed for the modeling process. stocks.drop(['Item'], axis=1) removes the 'Item' column. The axis=1 argument specifies that a column is being dropped (as opposed to a row, which would be axis=0).

~ **Importing Libraries:** Provides necessary tools for model building and evaluation.

~ **Normalization:** Ensures that all features are on a similar scale for better model performance.

~ **Dropping 'Item' Column:** Removes irrelevant data to focus on useful features for prediction.

By following these steps, the program is set up to preprocess the stock market data effectively, ensuring it is ready for accurate and efficient predictive modeling using neural networks.

From here the training n testing of data set is being started : 

X = stocks_data
# .drop(['Item'], axis=1)  # Replace 'TargetColumn' with the actual name of your target column
y = stocks_data['Volume']  # Replace 'TargetColumn' with the actual name of your target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the unroll function
def unroll(data, unroll_length):
    result = []
    for i in range(len(data) - unroll_length + 1):
        result.append(data[i:i+unroll_length])
    return np.array(result)

# Unroll the data
unroll_length = 50
X_train_unrolled = unroll(X_train.values, unroll_length)
X_test_unrolled = unroll(X_test.values, unroll_length)
y_train = y_train.values[unroll_length-1:]
y_test = y_test.values[unroll_length-1:]

print("X_train_unrolled", X_train_unrolled.shape)
print("y_train", y_train.shape)
print("X_test_unrolled", X_test_unrolled.shape)
print("y_test", y_test.shape)

The provided code is part of a larger program aimed at preparing stock market data for training a machine learning model, specifically an LSTM neural network. The code performs data splitting, reshaping (or "unrolling"), and printing the shapes of the resulting arrays. Let's break down each part and its purpose:

# 1.Setting up Features and Target:-
 X = stocks_data
 y = stocks_data['Volume'] 

~ X is set to the entire stocks_data DataFrame, and y is set to the 'Volume' column of stocks_data. The aim is to predict the 'Volume' of stocks. X contains all the features, while y is the target variable ('Volume') that the model will learn to predict.

# 2. Splitting Data:- 
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

~ This line splits the data into training and testing sets using an 80/20 split. shuffle=False ensures that the data is split in a time series order. training and testing datasets, allowing the model to be trained on one set of data and evaluated on another, ensuring that the evaluation metrics are unbiased.

# 3. Defining the Unroll Function:- 
 def unroll(data, unroll_length):
    result = []
    for i in range(len(data) - unroll_length + 1):
        result.append(data[i:i+unroll_length])
    return np.array(result)

~  This function takes in a dataset (data) and an unroll_length. It creates overlapping sequences of length unroll_length from the dataset. The unroll function is used to transform the data into a suitable format for training an LSTM model. LSTMs require input data to be in sequences of a fixed length.

# 4. Unrolling the Data:-
 unroll_length = 50
X_train_unrolled = unroll(X_train.values, unroll_length)
X_test_unrolled = unroll(X_test.values, unroll_length)
y_train = y_train.values[unroll_length-1:]
y_test = y_test.values[unroll_length-1:] 

~ These lines use the unroll function to reshape the training and testing datasets into sequences of length unroll_length (50 in this case). The target arrays (y_train and y_test) are also adjusted to match the length of the unrolled data. 

# 5. Printing Shapes of Unrolled Data:- 
 print("X_train_unrolled", X_train_unrolled.shape)
print("y_train", y_train.shape)
print("X_test_unrolled", X_test_unrolled.shape)
print("y_test", y_test.shape)
  
~ These lines print the shapes of the unrolled training and testing datasets and their corresponding target arrays. To verify that the data has been correctly reshaped and that the dimensions match the expected input and output sizes for the LSTM model. 

The outcome will be like this as shown below - 
 X_train_unrolled (2467, 50, 3)
 y_train (2467,)
 X_test_unrolled (580, 50, 3)
 y_test (580,)

*next step 
model = Sequential()
model.add(LSTM(50, input_shape=(unroll_length, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[mean_squared_error])

 The code defines and compiles a Sequential model using Keras, specifically designed for time series prediction with Long Short-Term Memory (LSTM) layers. Let's break down the key parts and understand their purposes:-

~Building the Model: Constructs a neural network model suitable for time series forecasting using LSTM layers to capture temporal dependencies.

~Preventing Overfitting: Incorporates Dropout to regularize the model and reduce overfitting.

~Predicting Continuous Values: Sets up a Dense layer with a linear activation for regression tasks.

~Configuring Training: Compiles the model with appropriate loss function, optimizer, and metrics to ensure effective training and evaluation.

Then this step is followed :- 

~ model.fit(X_train_unrolled, y_train, epochs=10, batch_size=32, validation_split=0.2) :- The line is used to train the LSTM model on the prepared training data.

# Here is step by step explination

1. model.fit():- The fit method is used to train the model on the given data. It starts the training process where the model learns from the training data.

2. X_train_unrolled:- This is the input data for training, which has been unrolled into sequences. Provides the features (input data) in the correct shape for the LSTM model.

3. y_train:- This is the target data for training. Provides the corresponding target values for the input sequences, which the model tries to predict.

4. epochs=10:- Specifies the number of complete passes through the training dataset. Determines how many times the model will see the entire training dataset. More epochs can lead to better learning but also increase the risk of overfitting if too high.

5. batch_size=32:- Defines the number of samples per gradient update. Determines how many samples the model processes before updating the weights. A smaller batch size generally leads to a more accurate estimate of the gradient but can be noisier.

6. validation_split=0.2:- Specifies the fraction of the training data to be used as validation data. The model uses 20% of the training data for validation to evaluate its performance on unseen data during training. This helps monitor overfitting and generalization.

# Model Building:

- **Sequential Model:**  The model is initialized as a sequential container for layers.

- **LSTM Layer:**  Adds an LSTM layer with 50 units, suitable for capturing temporal dependencies in sequential data. The input shape is defined by the unroll length and the number of features.

- **Dropout Layer:**  Adds a dropout layer to reduce overfitting by randomly setting a fraction of input units to 0 during training.

- **Dense Layer:**  Adds a dense layer with a single neuron for the output.

- **Activation Layer:**  Uses a linear activation function suitable for regression tasks.

- **Compilation:**  Compiles the model with mean squared error loss, Adam optimizer, and mean squared error metric for evaluation.

# Training the Model:

- **Input Data:**  X_train_unrolled (features) and y_train (target).

- **Epochs:** Set to 10, meaning the model will go through the entire dataset 10 times during training.

- **Batch Size:** Set to 32, determining how many samples the model processes before updating the weights.

- **Validation Split:** 20% of the training data is used for validation to monitor the model's performance on unseen data during training.

# Here is last step of the code :- 

# RMSE for Training-Testing Data with basic LSTM
trainScore = model.evaluate(X_train_unrolled, y_train, verbose=0)
train_mse = trainScore[1]  # mean_squared_error is the second element in the list
train_rmse = math.sqrt(train_mse)
print('Train Score: %.8f MSE (%.8f RMSE)' % (train_mse, train_rmse))

testScore = model.evaluate(X_test_unrolled, y_test, verbose=0)
test_mse = testScore[1]  # mean_squared_error is the second element in the list
test_rmse = math.sqrt(test_mse)
print('Test Score: %.8f MSE (%.8f RMSE)' % (test_mse, test_rmse))

The provided code calculates and prints the root mean squared error (RMSE) for the training and testing datasets. Let's break down its purpose and how it fits into the program:

# Model Evaluation:-
This code evaluates the performance of the LSTM model on both the training and testing datasets.
It calculates the mean squared error (MSE) for each dataset and then computes the square root of the MSE to obtain the RMSE, which provides a more interpretable measure of the error in the same units as the target variable.

# Calculating RMSE:-
The model.evaluate function returns the loss and metrics values, including MSE, for the specified data.
The second element of the returned list contains the MSE, which is then used to calculate RMSE.

# Printing Results:-
It prints the training and testing scores in terms of both MSE and RMSE.
Printing both MSE and RMSE gives a comprehensive understanding of the model's performance in terms of error.

# Assessing Model Performance:
By calculating and printing RMSE for both training and testing datasets, the code provides insights into how well the LSTM model generalizes to unseen data.
RMSE is a common metric for regression tasks, and lower values indicate better model performance.

By including these evaluation metrics in the program, you can gain insights into the effectiveness of the LSTM model for predicting stock volumes based on historical data.

# Final Outcome:

Train Score: 0.00063452 MSE (0.02518964 RMSE)
Test Score: 0.00020506 MSE (0.01431995 RMSE)

After training and evaluating the model, the final outcome is a set of predicted stock prices. These predictions are based on the patterns and relationships identified in the historical data. The model's performance metrics (MAE, MSE, RMSE) provide insight into the accuracy and reliability of the predictions.

This documentation provides a step-by-step guide to stock market analysis and prediction, ensuring even beginners can understand and implement the process.

# Conclusion:

Stock market prediction is complex due to market volatility. However, using historical data and deep learning models like LSTM networks with TensorFlow and Keras, we can make reasonably accurate predictions to assist in investment decisions. Continuous monitoring and model refinement are essential to maintain accuracy.

