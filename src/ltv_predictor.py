import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import glob
import re
from datetime import datetime

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras

# Use CPU only
tf.config.set_visible_devices([], 'GPU')

# Import required layers
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
Dropout = keras.layers.Dropout
Input = keras.layers.Input

class LTVPredictor:
   """LSTM-based model for LTV (Lifetime Value) prediction.
   
   This class implements an LSTM neural network to predict future LTV values
   based on historical data and growth rate patterns.
   """
   
   def __init__(self, base_path):
       """Initialize predictor with base path and model parameters.
       
       Args:
           base_path (str): Base directory containing data files
       """
       self.base_path = base_path
       self.model = None
       self.scaler = StandardScaler()
       self.min_growth_rate = 0.001  # Minimum growth rate threshold
       
   def load_historical_data(self):
       """Load and combine historical LTV data files.
       
       Returns:
           pd.DataFrame: Combined historical data
       
       Raises:
           ValueError: If no valid files found or data loading fails
       """
       files_dir = os.path.join(self.base_path, 'Files')
       all_files = os.listdir(files_dir)
       
       # Find files matching date pattern
       date_pattern = r'(\d{4}_\d{2}_\d{2})'
       matching_files = [
           os.path.join(files_dir, file) 
           for file in all_files 
           if bool(re.search(date_pattern, file))
           and not file.startswith('LTV_predictions_')
       ]
       
       if not matching_files:
           raise ValueError(f"No matching files found in {files_dir}")
       
       # Sort by date
       matching_files.sort(key=lambda x: datetime.strptime(
           re.search(date_pattern, os.path.basename(x)).group(1), 
           '%Y_%m_%d'
       ))
       
       # Load and combine data
       all_data = []
       for file in matching_files:
           try:
               df = pd.read_csv(file)
               date_str = re.search(date_pattern, os.path.basename(file)).group(1)
               df['file_date'] = pd.to_datetime(date_str, format='%Y_%m_%d')
               all_data.append(df)
               print(f"Successfully loaded: {os.path.basename(file)}")
           except Exception as e:
               print(f"Error loading {os.path.basename(file)}: {str(e)}")
       
       if not all_data:
           raise ValueError("No data could be loaded from the matching files")
       
       return pd.concat(all_data, ignore_index=True)

   def get_latest_data(self):
       """Retrieve most recent LTV analysis file.
       
       Returns:
           tuple: (DataFrame of latest data, path to latest file)
       """
       files_dir = os.path.join(self.base_path, 'Files')
       all_files = os.listdir(files_dir)
       
       # Find analysis files
       analysis_files = []
       for file in all_files:
           date_match = re.search(r'\d{4}_\d{2}_\d{2}', file)
           if date_match and file.endswith('.csv'):
               full_path = os.path.join(files_dir, file)
               analysis_files.append(full_path)
       
       print("Analysis files found:", [os.path.basename(f) for f in analysis_files])
       
       # Sort by date
       dated_files = []
       for file in analysis_files:
           date_match = re.search(r'(\d{4}_\d{2}_\d{2})', os.path.basename(file))
           if date_match:
               date_str = date_match.group(1)
               file_date = datetime.strptime(date_str, '%Y_%m_%d')
               dated_files.append((file, file_date))
       
       dated_files.sort(key=lambda x: x[1], reverse=True)
       latest_file = dated_files[0][0]
       
       return pd.read_csv(latest_file), latest_file

   def calculate_growth_rates(self, sequence):
       """Calculate daily growth rates from sequence.
       
       Args:
           sequence (np.array): Array of sequential values
           
       Returns:
           list: Calculated growth rates
       """
       growth_rates = []
       for i in range(len(sequence)-1):
           current, next_val = sequence[i:i+2]
           if current > 0:
               growth_rate = (next_val / current) - 1
           else:
               growth_rate = self.min_growth_rate
           growth_rates.append(max(growth_rate, self.min_growth_rate))
       return growth_rates

    def prepare_data(self, df, sequence_length=7):
     """Prepare data sequences for LSTM training.
     
     Args:
         df (pd.DataFrame): Input data frame
         sequence_length (int): Length of input sequences
         
     Returns:
         tuple: (X training data, y target data)
         
     Raises:
         ValueError: If no valid sequences can be created
     """
     days_cols = [f"day_{i}" for i in range(361)]
     df_days = df[days_cols]
     
     X, y = [], []
     for i in range(len(df_days)):
         for j in range(len(days_cols) - sequence_length - 1):
             sequence = df_days.iloc[i, j:j+sequence_length+1].values
             
             if not np.isnan(sequence).any() and all(sequence > 0):
                 growth_rates = self.calculate_growth_rates(sequence[:-1])
                 next_growth = (sequence[-1] / sequence[-2]) - 1 if sequence[-2] > 0 else self.min_growth_rate
                 next_growth = max(next_growth, self.min_growth_rate)
                 
                 X.append(growth_rates)
                 y.append(next_growth)
     
     X = np.array(X)
     y = np.array(y)
     
     if len(X) == 0 or len(y) == 0:
         raise ValueError("No valid sequences could be created from the data")
     
     X = X.reshape((X.shape[0], X.shape[1], 1))
     y = y.reshape(-1, 1)
     
     return X, y
  
  def build_model(self, sequence_length):
     """Build LSTM model architecture.
     
     Args:
         sequence_length (int): Length of input sequences
         
     Returns:
         keras.Model: Compiled LSTM model
     """
     model = Sequential([
         LSTM(32, input_shape=(sequence_length-1, 1), return_sequences=True),
         Dropout(0.2),
         LSTM(16),
         Dense(8, activation='relu'),
         Dense(1, activation='softplus')
     ])
     
     model.compile(
         optimizer='adam',
         loss='mse',
         metrics=['mae']
     )
     return model
  
  def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
     """Train the LSTM model.
     
     Args:
         X (np.array): Training data
         y (np.array): Target values
         validation_split (float): Validation set ratio
         epochs (int): Training epochs
         batch_size (int): Batch size
         
     Returns:
         tuple: (training history, test data tuple)
     """
     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=validation_split, random_state=42
     )
     
     self.model = self.build_model(X.shape[1]+1)
     
     history = self.model.fit(
         X_train, y_train,
         validation_data=(X_test, y_test),
         epochs=epochs,
         batch_size=batch_size,
         verbose=1
     )
     
     return history, (X_test, y_test)
  
  def predict_future_ltv(self, sequence_data, days_to_predict):
     """Predict future LTV values.
     
     Args:
         sequence_data (np.array): Initial sequence for prediction
         days_to_predict (int): Number of days to predict ahead
         
     Returns:
         list: Predicted LTV values
         
     Raises:
         ValueError: If model is not trained
     """
     if self.model is None:
         raise ValueError("Model has not been trained yet")
     
     base_value = sequence_data[-1]
     sequence = sequence_data.copy()
     
     growth_rates = self.calculate_growth_rates(sequence)
     current_sequence = np.array(growth_rates).reshape(1, -1, 1)
     
     predictions = [base_value]
     current_value = base_value
     
     for _ in range(days_to_predict):
         next_growth = float(self.model.predict(current_sequence, verbose=0)[0, 0])
         next_growth = max(next_growth, self.min_growth_rate)
         
         next_value = current_value * (1 + next_growth)
         next_value = max(next_value, current_value * (1 + self.min_growth_rate))
         predictions.append(next_value)
         
         current_sequence = np.roll(current_sequence, -1, axis=1)
         current_sequence[0, -1, 0] = next_growth
         current_value = next_value
     
     return predictions[1:]
  
  def process_and_predict(self, sequence_length=7, prediction_days=30, epochs=50):
     """Execute full prediction pipeline.
     
     Args:
         sequence_length (int): Training sequence length
         prediction_days (int): Days to predict
         epochs (int): Training epochs
         
     Returns:
         tuple: (predictions DataFrame, output file path)
     """
     print("Starting prediction process...")
     
     try:
         print("Loading historical data...")
         historical_data = self.load_historical_data()
         print(f"Loaded {len(historical_data)} historical records")
         
         print("Preparing training data...")
         X, y = self.prepare_data(historical_data, sequence_length)
         print(f"Created {len(X)} sequences for training")
         
         print("Training model...")
         history, (X_test, y_test) = self.train(X, y, epochs=epochs)
         print("Model training completed")
         
         print("Loading latest data...")
         latest_data, latest_file = self.get_latest_data()
         latest_date = re.search(r'(\d{4}_\d{2}_\d{2})', latest_file).group(1)
         
         print("Latest data shape:", latest_data.shape)
         days_cols = [f"day_{i}" for i in range(sequence_length)]
         
         if not all(col in latest_data.columns for col in days_cols):
             files_dir = os.path.join(self.base_path, 'Files')
             analysis_files = [
                 os.path.join(files_dir, file) 
                 for file in os.listdir(files_dir) 
                 if file.startswith('[MKT]_LTV_Analysis_') and file.endswith('.csv')
             ]
             analysis_files.sort(key=lambda x: datetime.strptime(
                 re.search(r'(\d{4}_\d{2}_\d{2})', os.path.basename(x)).group(1), 
                 '%Y_%m_%d'
             ))
             
             latest_analysis_file = analysis_files[-1]
             analysis_df = pd.read_csv(latest_analysis_file)
             
             for col in days_cols:
                 if col not in latest_data.columns:
                     latest_data[col] = analysis_df[col].iloc[0]
         
         predictions_df = pd.DataFrame()
         predictions_df['Date'] = latest_data['Date']
         
         print("Making predictions...")
         for idx in range(len(latest_data)):
             initial_sequence = latest_data.iloc[idx][days_cols].values
             
             if any(initial_sequence <= 0):
                 print(f"Invalid sequence at row {idx}")
                 continue
             
             try:
                 predictions = self.predict_future_ltv(initial_sequence, prediction_days)
                 
                 for day, pred in enumerate(predictions, start=sequence_length):
                     if day <= sequence_length + prediction_days:
                         col_name = f'P_D{day}'
                         predictions_df.loc[idx, col_name] = pred
                 
                 print(f"Processed row {idx}")
             except Exception as e:
                 print(f"Error at row {idx}: {str(e)}")
             
             if (idx + 1) % 10 == 0:
                 print(f"Completed {idx + 1}/{len(latest_data)} rows")
         
         if predictions_df.empty:
             raise ValueError("No predictions could be generated")
         
         result = predictions_df.sort_values(by="Date", ascending=True)
         
         pred_cols = [col for col in result.columns if col.startswith('P_D')]
         for col in pred_cols:
             result[col] = pd.to_numeric(result[col], errors='coerce').round(2)
         
         output_filename = f'LTV_predictions_epochs{epochs}_{latest_date}.csv'
         predictions_dir = os.path.join(self.base_path, 'Files_Predictions')
         
         if not os.path.exists(predictions_dir):
             os.makedirs(predictions_dir)
             
         output_path = os.path.join(predictions_dir, output_filename)
         result.to_csv(output_path, index=False, float_format='%.2f')
         print(f"Predictions saved to: {output_path}")
         
         return result, output_path
         
     except Exception as e:
         print(f"An error occurred: {str(e)}")
         raise

if __name__ == "__main__":
   base_path = '/Users/yopa/Desktop/LTV_ML'
   predictor = LTVPredictor(base_path)
