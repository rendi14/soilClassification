from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_ready = pd.read_csv("dataset/features_extracted/data_extracted_full_grayscale_resized.csv")
class_mapping = {
    "Aluvial": 0,
    "Andosol": 1,
    "Entisol": 2,
    "Humus": 3,
    "Inceptisol": 4,
    "Kapur": 5,
    "Laterite": 6,
    "Pasir": 7
}

df_ready['label'] = df_ready['label'].map(class_mapping)
df_ready

X = np.array(df_ready[['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']])
y = np.array(df_ready['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scalers/scaler_yang_dipake.save')
print("Scaler has been saved!")
