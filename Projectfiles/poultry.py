import os
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import cv2
base_dir = 'dataset'
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)


base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)


os.makedirs("model", exist_ok=True)
model.save("model/poultry_model.h5")

val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=list(val_generator.class_indices.keys())))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

model = load_model("model/poultry_model.h5")

img_path = "C:\\Users\\User\\PYTHON\\dataset"
img = cv2.imread(img_path)
img = cv2.resize(img, (img_width, img_height))
img = img / 255.0
img = np.expand_dims(img, axis=0) 

prediction = model.predict(img)
predicted_class = np.argmax(prediction)


class_labels = list(train_generator.class_indices.keys())
print("Predicted Class:", class_labels[predicted_class])
