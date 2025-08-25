# 📌 Step 1: Install required packages
# Run this in your terminal before executing the script:
# pip install librosa numpy scikit-learn tensorflow tqdm matplotlib

# 📌 Step 2: Import libraries
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape

# 📌 Step 3: Define paths and emotion map
dataset_path = './ravdess'  # Update this to your local folder
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# 📌 Step 4: Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 📌 Step 5: Load data
data, labels = [], []
for file in tqdm(os.listdir(dataset_path)):
    if file.endswith('.wav'):
        emotion_code = file.split('-')[2]
        emotion = emotion_map.get(emotion_code)
        feature = extract_features(os.path.join(dataset_path, file))
        data.append(feature)
        labels.append(emotion)

# 📌 Step 6: Prepare dataset
X = np.array(data)
y = LabelEncoder().fit_transform(labels)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 7: Build LSTM model
model = Sequential([
    Reshape((40, 1), input_shape=(40,)),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 📌 Step 8: Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 📌 Step 9: Plot accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()