import matplotlib.pyplot as plt
import seaborn as sns

# 히트맵 생성
plt.figure(figsize=(12, 8))
sns.heatmap(
    day_stats.pivot_table(values='mean', index='Day'),
    cmap='RdYlBu',
    center=0,
    annot=True
)
plt.title('LTV Prediction Accuracy by Day')
plt.xlabel('Prediction Day')
plt.ylabel('Accuracy')
plt.savefig('results/prediction_accuracy.png')
plt.close()

# MAE 추세 그래프
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig('results/mae_trend.png')
plt.close()
