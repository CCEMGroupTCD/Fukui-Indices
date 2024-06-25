import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import numpy as np


# Extract 'normalised_Fukui_DFT' as the x-values
file_path = '../data/output.csv'
data = pd.read_csv(Path(file_path).resolve())

## Mantings code:
data.columns = data.iloc[0]
data = data[1:]
data.columns = [str(col).strip() for col in data.columns]
x_values = data['normalised_Fukui_DFT'].astype(float).tolist()
# Initialize lists to hold x and y values
y_values = []
aligned_x_values = []
aligned_y_values = []
molecule_names = []
smiles = []
# Loop through columns CV_0 to CV_30
for i in range(31):
    cv_column = f'CV_{i}'
    pred_column = f'pred_normalised_Fukui_DFT_RF_CV_{i}'
    # Ensure columns exist
    if cv_column in data.columns and pred_column in data.columns:
        for index, row in data.iterrows():
            if row[cv_column] == 'test':
                aligned_x_values.append(float(row['normalised_Fukui_DFT']))
                aligned_y_values.append(float(row[pred_column]))
                molecule_names.append(row['molecule_name'])
                smiles.append(row['smiles'])


print(f'MAE: {mean_absolute_error(aligned_x_values, aligned_y_values)}')
print(f'R2: {r2_score(aligned_x_values, aligned_y_values)}')

# Use pandas to calculate the accuracy for each group
df = pd.DataFrame({'x': aligned_x_values, 'y': aligned_y_values, 'smiles': smiles, 'molecule_name': molecule_names})
accuracies = df.groupby('molecule_name').apply(lambda group: np.array(np.argsort(group['x']) == np.argsort(group['y']))[-1]).to_frame().rename(columns={0: 'correct'})
print(f'Accuracy: {accuracies["correct"].mean()}')
# Add smiles to accuracies
accuracies = accuracies.join(df.groupby('molecule_name').first()['smiles'])
accuracies.to_csv('accuracies.csv')


plt.figure(figsize=(10, 6))
plt.scatter(aligned_x_values, aligned_y_values, alpha=0.2)
plt.title('Scatter Plot of Normalised Fukui DFT vs Predicted Values')
plt.xlabel('Normalised Fukui DFT')
plt.ylabel('Predicted Normalised Fukui DFT RF')
plt.grid(True)
plt.show()