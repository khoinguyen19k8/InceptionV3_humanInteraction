from keras import activations
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from inceptionv3_highfive_model import model_inceptionv3
from sklearn.model_selection import train_test_split
import os
from skimage import exposure, io, data, img_as_float
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from tempfile import mktemp
import time
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import tensorflow as tf
import sys

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

if len(sys.argv) != 2:
    raise Exception("Your arguments are incorrect!")

POS_VAL = sys.argv[1]
DATA_PATH = os.path.join("tv_human_interactions_videos", f"frames_pos_{POS_VAL}")
if not os.isdir(DATA_PATH):
    raise Exception("Wrong frame value!")
# load images
fileslist = []
number_classes = 0
for classes in os.listdir(DATA_PATH):
    number_classes = number_classes + 1
    sd = os.path.join(DATA_PATH, classes)
    for files in os.listdir(sd):
        fileslist.append(sd + files)

np.random.shuffle(fileslist)

classes = []
X = []
i = 1
length = len(fileslist)
for f in fileslist:
    img_class = int((f.split("/")[-1]).split("_")[0])
    # Default size for Inception is 299x299
    img = image.load_img(f, target_size=(299, 299))
    print("Processed: " + str(i / length))
    img_h = image.img_to_array(img)
    img_h /= 255
    X.append(img_h)
    classes.append(img_class)
    i = i + 1


X = np.array(X, dtype="float32")
Y = np.eye(number_classes, dtype="uint8")[classes]

x_train, x_valtest, y_train, y_valtest = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_valtest, y_valtest, test_size=0.5, random_state=42
)

K.clear_session()

# Load model
model = model_inceptionv3(299, 299, 3, number_classes)


# training
batch_size = 16
nb_epochs = 100

# datagenerator based on image augmentation
datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

datagen.fit(x_train)


early_stopping_monitor = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=12, verbose=0, mode="min"
)

# Initialize weight and model path

weight_and_model_path = os.path.join("weights-and-models", f"frame_val_{POS_VAL}")
weight_path = os.path.join(weight_and_model_path, "weights.best.hdf5")
model_path = os.path.join(weight_and_model_path, "model.json")

checkpoint = ModelCheckpoint(
    weight_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
callbacks_list = [checkpoint, early_stopping_monitor]


history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=nb_epochs,
    callbacks=callbacks_list,
    shuffle=True,
    validation_data=(x_val, y_val),
)
score = model.evaluate(x_val, y_val, verbose=0)

print("Final score:", score[0])
print("Final accuracy:", score[1])
print("EVERYTHING: ", score)

# serialise model to json
json_string = model.to_json()
with open(model_path, "w") as json_file:
    json_file.write(json_string)

# Predictions
eval = model.evaluate(x_test, y_test, verbose=0)
print("------------------------------")
print("Test score:", eval[0])
print("Test accuracy:", eval[1])
print("Everyting: ", eval)
print("------------------------------")
