import tensorflow as tf
from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image


train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                     image_size=(128, 128), batch_size=30, label_mode="categorical")
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                     image_size=(128, 128), batch_size=30, label_mode="categorical")


# нормалізація зображень від 0 до 1
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#побудова моделі
model = models.Sequential()

model.add(layers.Conv2D(filters = 32, kernel_size = (3,3),
                        activation='relu', input_shape=(128, 128, 3))) #32 найпростіші ознаки
model.add(layers.MaxPooling2D(pool_size=(2,2)))# зменншили карту ознак у 2 рази


model.add(layers.Conv2D(filters = 64, kernel_size = (3,3),
                        activation='relu'))#конутіри + структури
model.add(layers.MaxPooling2D(pool_size=(2,2)))


model.add(layers.Conv2D(filters = 128, kernel_size = (3,3),
                        activation='relu'))#конутіри + структури + форми об'єктів
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())#vektor

#vnytrishni shar
model.add(layers.Dense(64, activation='relu'))

#test shar
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#navchanna

histore = model.fit(
    train_ds,
    epochs = 15,
    validation_data=test_ds
)


#ocinka
test_lost, test_acc = model.evaluate(test_ds)
print('Yakist:', test_acc)

#perevirka

class_name = ['cars', 'cats', 'dogs']
img = image.load_img('images/', target_size=(128, 128))
img_array = image.img_to_array(img)

img_array = img_array/255.0
img_array = np.expand_dims(img_array, axis=0)
preditions = model.predict(img_array)

predicted_index = np.argmax(preditions[0])
print(f'Шьщвірнісь по класам: {preditions[0]}')
print(f'Модель визначила: {class_name[predicted_index]}')