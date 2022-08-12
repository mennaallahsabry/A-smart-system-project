import os

import speech_recognition as sr
import pyaudio
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from tensorflow.keras import datasets, layers, models

recognizer = sr.Recognizer()

''' recording the sound '''

with sr.Microphone() as source:
    print("Adjusting noise ")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Recording for 4 seconds")
    print("Speak Now")
    recorded_audio = recognizer.listen(source, timeout=4)
    print("Done recording")

''' Recorgnizing the Audio '''
try:
     print("Recognizing the text")
     text = recognizer.recognize_google(recorded_audio, language="en-US")

     print("Decoded Text : {}".format(text))

     (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
     training_images, testing_images = training_images / 255, testing_images / 255

     class_names = ['plane', ' car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

     for i in range(16):
        plt.subplot(4, 4, i + 1)

        plt.xticks([])
        plt.yticks([])
        plt.imshow(training_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[training_labels[i][0]])

     plt.show()

     training_images = training_images[:20000]
     training_labels = training_labels[:20000]
     testing_images = testing_images[:4000]
     testing_labels = testing_labels[:4000]

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
#
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")
#
# model.save('image_classifier.model')
     model = models.load_model('image_classifier.model')

     images = ['0x0-horses-the-wings-of-mankind-1527015927739', '2022_Acura_ILX_1', '4032687', 'how heavy is semi truck',
          'Microsoft_Flight_Simulator_beginners', 'resized_250499-1b-deer-0618_85-26607_t600']
     for img in images:
        img = "\\" + img + ".jpg"

        imgs = cv.imread(r'C:\Users\AUC\PycharmProjects\pythonProject2\Image data' + img)
        imgs = cv.cvtColor(imgs, cv.COLOR_BGR2RGB)
        plt.imshow(imgs, cmap=plt.cm.binary)
        prediction = model.predict(np.array([imgs]) / 255)
        index = np.argmax(prediction)

        if recognizer.recognize_google(recorded_audio, language="en-US") == class_names[index]:
            print(f'prediction is {class_names[index]}')
            plt.show()
except Exception as ex:
    print(ex)
