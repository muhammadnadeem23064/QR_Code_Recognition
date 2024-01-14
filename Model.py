import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2


# import model and initialize the number of training epochs and batch size
from tensorflow.keras.applications import MobileNetV2 as XModel

num_epochs = 50
batch_size = 32
input_size = 224
num_classes = 15
lr= 0.0001


# derive the path to the directories containing the training,
# validation, and testing splits, respectively
TRAIN_PATH = os.path.sep.join(["/data2/arai/SCI/dataset/QR_VGG/Pattern1_Finder/new_model", "training"])
VAL_PATH = os.path.sep.join(["/data2/arai/SCI/dataset/QR_VGG/Pattern1_Finder/new_model", "validation"])
TEST_PATH = os.path.sep.join(["/data2/arai/SCI/dataset/QR_VGG/Pattern1_Finder/new_model", "testing"])


# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(os.listdir(TRAIN_PATH))
totalVal = len(os.listdir(VAL_PATH))
totalTest = len(os.listdir(TEST_PATH))


#totalTrain = len(TRAIN_PATH)len(list(paths.list_images(TRAIN_PATH)))
#totalVal = len(list(paths.list_images(VAL_PATH)))
#totalTest = len(list(paths.list_images(TEST_PATH)))
totalTrain
totalVal
totalTest


# initialize the training training data augmentation object
trainAug=ImageDataGenerator(
rotation_range=20,
zoom_range=0.01,
width_shift_range=0.01,
height_shift_range=0.01,
shear_range=0.01,
horizontal_flip=True,
vertical_flip=True,
fill_mode="nearest")


# initialize the validation  data augmentation object                                                
valAug=ImageDataGenerator()
# initialize the  testing data augmentation object
testAug=ImageDataGenerator()


# initialize the training generator
trainGen=trainAug.flow_from_directory(
                                                 TRAIN_PATH,
                                                 target_size=(input_size, input_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)

# initialize the validation generator
valGen = valAug.flow_from_directory(
                                                 VAL_PATH,
                                                 target_size=(input_size, input_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=False)

# initialize the testing generator
testGen=testAug.flow_from_directory(
                                                 TEST_PATH,
                                                 target_size=(input_size, input_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=False)


basemodel = XModel(include_top=False, weights='imagenet')
model = Sequential()
model.add(basemodel)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.03)))
model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.03)))

model.add(Dense(num_classes, activation="softmax"))
model.summary()
opt=Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])


a = datetime.datetime.now()
history = model.fit_generator(
        trainGen,
        steps_per_epoch=len(trainGen),
        validation_data=valGen,
        validation_steps=len(valGen),
        epochs=num_epochs)


#======== Result Plotting ========#
#======== Accuracy Plot ========#
print(history.history.keys())
plt.plot(history.history['accuracy'], 'o-')
plt.plot(history.history['val_accuracy'], 'x-')
plt.title('MobileNetV1')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc=0)
plt.savefig("/data2/arai/SCI/dataset/QR_VGG/Pattern1_Finder/new_model/Results/MobileNetV1Accuracy.png")
plt.show()

#======== Loss Plot ========#
plt.plot(history.history['loss'], 'o-')
plt.plot(history.history['val_loss'], 'x-')
plt.title('MobileNetV1')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc=0)
plt.savefig("/data2/arai/SCI/dataset/QR_VGG/Pattern1_Finder/new_model/Results/MobileNetV1Loss.png")
plt.show()

y_pred=model.predict(testGen)
y_pred=np.argmax(y_pred, axis=1)
target_names = ["RI01","RI02","RI03", "RI04","RI05","RI06", "RI07","RI08", "RI09","RI10","RI11", "RI12","RI13","RI14", "RI15"]

#======== Confusion Matrix ========#
cm = confusion_matrix(y_pred, testGen.classes)
print("***** Confusion Matrix *****")
print(cm)
print("***** Classification Report *****")
print(classification_report(y_pred, testGen.classes ,target_names=target_names))
classes= num_classes

model.save("/data2/arai/SCI/dataset/QR_VGG/Pattern1_Finder/new_model/Results/MobileNetv1.h5")


loss, accuracy = model.evaluate(testGen)
print("Test Loss: {:.2f}".format(loss))
print("Test Accuracy: {:.2f}%".format(accuracy * 100))