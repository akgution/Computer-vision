import pandas as pd # таблиці
import numpy as np
import tensorflow as tf # для нейронок
from tensorflow import keras
from tensorflow.keras import layers # шари
from sklearn.preprocessing import LabelEncoder # перероблює текст в числа
import matplotlib.pyplot as plt # побудова графіків

# 2.робота з ссв файлом
df = pd.read_csv('data/figures_full.csv')
# print(df.head())
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# 3.обрали елементи для навчання
X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

# 4.створення моделі
model = keras.Sequential([layers.Dense(16, activation='relu', input_shape=(3, )),
                          layers.Dense(16, activation='relu'),
                          layers.Dense(16, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 5.навчання
history = model.fit(X, y, epochs = 300, verbose = 0)

# 6.візуалізація навчання
plt.plot(history.history['loss'], label = "Втрата")
plt.plot(history.history['accuracy'], label = "Точність")
plt.xlabel('епоха')
plt.ylabel('значення')
plt.title("процес навчання")
plt.legend()
plt.show()

# 7.тестування
test = np.array([[48, 42, 6]])

pred = model.predict(test)

print(f'Іймовірність кожного класу: {pred}')
print(f'модель визначила: {encoder.inverse_transform([np.argmax(pred)])[0]}')