import os
from ltv_predictor import LTVPredictor
from ltv_comparator import LTVComparator

def run_ltv_analysis(base_path, prediction_days=30, epochs=50):
   """Run complete LTV prediction and correction pipeline.
   
   Args:
       base_path (str): Base directory for data files
       prediction_days (int): Number of days to predict ahead
       epochs (int): Number of training epochs
       
   Returns:
       tuple: (Final predictions DataFrame, Differences statistics DataFrame)
   """
   
   # Step 1: Initial LTV Prediction
   print("\n=== Starting LTV Prediction ===")
   predictor = LTVPredictor(base_path)
   predictions_df, _ = predictor.process_and_predict(
       prediction_days=prediction_days,
       epochs=epochs
   )
   
   # Step 2: Prediction Correction
   print("\n=== Starting Prediction Comparison and Correction ===")
   comparator = LTVComparator(base_path, prediction_days=prediction_days)
   final_predictions, differences_df = comparator.process_and_save()
   
   # Step 3: Generate Statistics
   print("\n=== Prediction Statistics ===")
   day_stats = differences_df.groupby('Day')['Difference_Percent'].agg(['mean', 'std', 'count'])
   print("\nDifference Statistics by Day:")
   print(day_stats.round(4))
   
   return final_predictions, differences_df

if __name__ == "__main__":
   base_path = '/Users/yopa/Desktop/LTV_ML'
   prediction_days = 30
   epochs = 50
   
   try:
       final_predictions, differences_df = run_ltv_analysis(
           base_path,
           prediction_days=prediction_days,
           epochs=epochs
       )
       print("\nProcess completed successfully!")
   except Exception as e:
       print(f"\nProcess failed: {str(e)}")
