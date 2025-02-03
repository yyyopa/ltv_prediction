import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

class LTVComparator:
    def __init__(self, base_path, prediction_days=30):
        self.base_path = base_path
        self.prediction_days = prediction_days

    def load_files(self):
        # Find the most recent original file
        files_dir = os.path.join(self.base_path, 'Files')
        analysis_files = [f for f in os.listdir(files_dir) if '[MKT]_LTV_Analysis_' in f]
        latest_file = max(analysis_files)
        original_file = os.path.join(files_dir, latest_file)

        # Find the most recent prediction file
        pred_dir = os.path.join(self.base_path, 'Files_Predictions')
        pred_files = [f for f in os.listdir(pred_dir) if 'LTV_predictions' in f]
        self.latest_pred = max(pred_files)  # Store as class variable
        prediction_file = os.path.join(pred_dir, self.latest_pred)
        
        return pd.read_csv(original_file), pd.read_csv(prediction_file)
    
    def get_max_day_for_date(self, row):
        """Find the maximum available day column for each date"""
        day_columns = [col for col in row.index if col.startswith('day_')]
        available_days = [col for col in day_columns if not pd.isna(row[col]) and row[col] > 0]
        return int(max(available_days).replace('day_', '')) if available_days else 0
    
    def calculate_weekly_differences(self, original_df, prediction_df):
        """Calculate differences between actual and predicted values by week"""
        differences = []
        
        # Maintain original date format
        original_date = original_df['Date'].copy()
        prediction_date = prediction_df['Date'].copy()
        
        # Temporarily use start date for calculations
        original_df['Date'] = original_df['Date'].apply(lambda x: x.split('~')[0].strip())
        prediction_df['Date'] = prediction_df['Date'].apply(lambda x: x.split('~')[0].strip())
        
        # Convert Date to datetime 
        original_df['Date'] = pd.to_datetime(original_df['Date'])
        prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])
        
        # Find maximum day for each row
        original_df['max_day'] = original_df.apply(self.get_max_day_for_date, axis=1)
        
        for _, row in original_df.iterrows():
            max_day = row['max_day']
            if max_day < 7:  # Require at least 7 days of data
                continue
                
            date = row['Date']
            pred_row = prediction_df[prediction_df['Date'] == date]
            
            # Handle cases with no matching prediction
            if pred_row.empty:
                print(f"No matching prediction found for date: {date}")
                continue
                
            pred_row = pred_row.iloc[0]
            
            # Calculate differences within available day range
            for d in range(7, min(max_day + 1, self.prediction_days + 7)):
                actual_val = row[f'day_{d}']
                pred_val = pred_row[f'P_D{d}']
                
                if not pd.isna(actual_val) and not pd.isna(pred_val) and actual_val > 0:
                    diff_percent = (actual_val - pred_val) / pred_val
                    differences.append({
                        'Date': date,
                        'Day': d,
                        'Actual': actual_val,
                        'Predicted': pred_val,
                        'Difference_Percent': diff_percent
                    })
        
        differences_df = pd.DataFrame(differences)
        
        # Restore original date format
        differences_df['Date'] = differences_df['Date'].map(dict(zip(
            pd.to_datetime(original_df['Date'].unique()), 
            original_date.unique()
        )))
        
        return differences_df
    
    def apply_corrections(self, prediction_df, differences_df):
        """Calculate final predictions by learning from differences"""
        final_predictions = prediction_df.copy()
        
        # Calculate average differences by day
        avg_differences = differences_df.groupby('Day')['Difference_Percent'].mean()
        
        # Adjust predictions
        for day in range(7, self.prediction_days + 7):
            if day in avg_differences.index:
                col_name = f'P_D{day}'
                if col_name in final_predictions.columns:
                    correction_factor = 1 + avg_differences[day]
                    final_predictions[col_name] = final_predictions[col_name] * correction_factor
        
        return final_predictions.round(2)
    
    def process_and_save(self):
        print("Starting the comparison process...")
        
        try:
            # Load files
            original_df, prediction_df = self.load_files()
            print("Files loaded successfully")
            
            # Calculate differences
            print("Calculating differences...")
            differences_df = self.calculate_weekly_differences(original_df, prediction_df)
            print(f"Found {len(differences_df)} comparison points")
            
            # Calculate final predictions
            print("Applying corrections...")
            final_predictions = self.apply_corrections(prediction_df, differences_df)
            
            # Create Files_Predictions_Final directory
            final_predictions_dir = os.path.join(self.base_path, 'Files_Predictions_Final')
            if not os.path.exists(final_predictions_dir):
                os.makedirs(final_predictions_dir)
                print(f"Created directory: {final_predictions_dir}")
            
            # Save results
            latest_date = re.search(r'(\d{4}_\d{2}_\d{2})', self.latest_pred).group(1)
            output_path = os.path.join(self.base_path, 'Files_Predictions_Final', f'LTV_predictions_final_{latest_date}.csv')
            final_predictions.to_csv(output_path, index=False)
            print(f"Final predictions saved to: {output_path}")
            
            return final_predictions, differences_df
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

if __name__ == "__main__":
    base_path = '/Users/yopa/Desktop/LTV_ML'
    comparator = LTVComparator(base_path)
