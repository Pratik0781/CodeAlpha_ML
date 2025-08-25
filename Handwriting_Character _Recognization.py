# file: mnist_cnn.py
import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1) Reproducibility ---
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- 2) Load & preprocess MNIST (28x28 grayscale digits 0–9) ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# scale to [0,1] and add channel dim
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test  = (x_test.astype("float32")  / 255.0)[..., None]

# train/val split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
)

# --- 3) Model (lightweight CNN) ---
def build_model(num_classes=10):
    inputs = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="mnist_cnn")

model = build_model()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# --- 4) Train ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )
]
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=15, batch_size=128,
    callbacks=callbacks, verbose=2
)

# --- 5) Evaluate ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Confusion matrix & report
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)

# --- 6) Quick visualization ---
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()

# Show a few predictions
def show_examples(images, y_true, y_hat, n=6):
    idx = np.random.choice(len(images), n, replace=False)
    plt.figure(figsize=(10,2))
    for i, k in enumerate(idx):
        plt.subplot(1,n,i+1)
        plt.imshow(images[k].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(f"T:{y_true[k]} P:{y_hat[k]}")
    plt.tight_layout(); plt.show()

show_examples(x_test, y_test, y_pred, n=8)

# --- 7) Save model ---
os.makedirs("models", exist_ok=True)
model.save("models/mnist_cnn.h5")
print("Saved model to models/mnist_cnn.h5")
