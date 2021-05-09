import tensorflow as tf
from keras.preprocessing import image
import os
import numpy as np
from sklearn.model_selection import train_test_split
pos_list = list(range(2, 11))

for pos_val in pos_list:
    print(f"Start processing value {pos_val}")
    DATA_PATH = os.path.join("tv_human_interactions_videos", f"frames_pos_{pos_val}")
    ARRAY_PATH = os.path.join("preprocessed-arrays", f"frames_pos_{pos_val}")

    if not os.path.isdir(
        ARRAY_PATH
    ):  # Creating a folder for arrays if it does not exist.
        os.mkdir(ARRAY_PATH)

    fileslist = []
    number_classes = 0
    for classes in os.listdir(DATA_PATH):
        number_classes = number_classes + 1
        sd = os.path.join(DATA_PATH, classes)
        for files in os.listdir(sd):
            fileslist.append(os.path.join(sd, files))

    np.random.shuffle(fileslist)

    classes = []
    X = []
    i = 1
    length = len(fileslist)
    for f in fileslist:
        
        raw_f = r"{}".format(f)
        img_class = int((raw_f.split(os.sep)[-1]).split("_")[0])
        # Default size for Inception is 299x299
        img = image.load_img(f, target_size=(299, 299))
        if i % 1000 == 0:
            print("Processed: " + str(i / length))
        img_h = image.img_to_array(img)
        img_h /= 255
        X.append(img_h)
        classes.append(img_class)
        i = i + 1

    X = np.array(X, dtype="float32")
    Y = np.eye(number_classes, dtype="uint8")[classes]  # One-hot coding

    if os.path.isdir(os.path.join(ARRAY_PATH, "X.npy")) or os.path.isdir(
        os.path.join(ARRAY_PATH, "Y.npy")
    ):
        raise Exception("Arrays already existed!")

    x_train, x_valtest, y_train, y_valtest = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_valtest, y_valtest, test_size=0.5, random_state=42
    )

    np.save(os.path.join(ARRAY_PATH, "X.npy"), X)
    np.save(os.path.join(ARRAY_PATH, "Y.npy"), Y)

    np.save(os.path.join(ARRAY_PATH, "x_train.npy"), x_train)
    np.save(os.path.join(ARRAY_PATH, "x_val.npy"), x_val)
    np.save(os.path.join(ARRAY_PATH, "x_test.npy"), x_test)

    np.save(os.path.join(ARRAY_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(ARRAY_PATH, "y_val.npy"), y_val)
    np.save(os.path.join(ARRAY_PATH, "y_test.npy"), y_test)

    print("classes shape: ", np.shape(classes))
    print("Y shape: ", Y.shape)
    print(f"Done processing value {pos_val}")
