import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Definir rutas a los datos
train_dir = "C:\\Users\\59598\\Desktop\\Simeio\\Data\\train"
validation_dir = "C:\\Users\\59598\\Desktop\\Simeio\\Data\\validation"
test_dir = "C:\\Users\\59598\\Desktop\\Simeio\\Data\\train"

# Definir parámetros de preprocesamiento y generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode='categorical')

# Asegurarse de que train_generator genere al menos un lote de datos
num_train_samples = len(train_generator.filenames)
batch_size = train_generator.batch_size
steps_per_epoch = np.ceil(num_train_samples / batch_size)

print("Número total de muestras de entrenamiento:", num_train_samples)
print("Tamaño del lote:", batch_size)
print("Pasos por época:", steps_per_epoch)

# Imprimir la lista de archivos cargados en el generador de datos
print("Archivos de entrenamiento cargados:")
for filename in train_generator.filenames:
    print(filename)

# Verificar si hay datos cargados en el generador de datos
if num_train_samples == 0:
    print("No se han cargado muestras de entrenamiento. Por favor, verifica la ruta del directorio de entrenamiento.")
    exit()

# Contar el número de clases en el conjunto de datos de entrenamiento
num_classes = len(train_generator.class_indices)
print("Número de clases:", num_classes)
# Cargar el modelo base preentrenado
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Agregar capas adicionales al modelo base
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Definir el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=30,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))

# Guardar el modelo entrenado
model.save("C:\\Users\\59598\\Desktop\\Simeio\\Model\\keras_model.keras")