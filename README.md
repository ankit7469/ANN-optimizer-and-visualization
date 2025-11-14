
#  Artificial Neural Network (ANN) — Optimizer & Training Visualization

This project demonstrates a simple **Artificial Neural Network (ANN)** built using **TensorFlow + Keras**.  
It also shows how different components like **optimizer**, **activation functions**, and **loss visualization** affect model performance.

---

##  Project Highlights

- Built a fully functional **ANN model** using Keras  
- Used **Adam optimizer** for efficient learning  
- Visualized **training loss & accuracy**  
- Predicted outputs on example inputs  
- Simple and beginner-friendly deep learning project  

---

##  What I Learned

### ✔ Artificial Neural Networks (ANN)
- ANN is a collection of connected artificial neurons.
- Used for regression & classification.
- Input → Hidden Layers → Output.

### ✔ Optimizer (Adam)
- Adam is a fast and adaptive optimizer.
- Automatically adjusts learning rate.
- Helps the model converge faster.

### ✔ Activation Functions
- **ReLU** → Hidden layers  
- **Sigmoid** → Output layer (for binary classification)

### ✔ Loss & Accuracy Visualization
- Plotted loss/accuracy across epochs  
- Helps understand model learning progress  
- Identifies overfitting or underfitting  

---

## 🧾 Tech Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

## 💻 Code Overview

```python
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=20)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(["Loss", "Accuracy"])
plt.show()
