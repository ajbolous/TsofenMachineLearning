from model import buildModel
import numpy as np
import tflearn
import matplotlib.pyplot as plt

model = buildModel()
model.load('mnist.model')

images, labels, testImages, testLabels = tflearn.datasets.mnist.load_data(one_hot=True)

for i in range(1,10):
    
    raw_prediction = model.predict([images[i]])[0]
    print ("Predicted ", raw_prediction)

    digit = np.argmax(raw_prediction)
    print ("Predicted digit is : ", digit)

    plt.imshow(images[i].reshape((28,28)), cmap='gray')
    plt.show()
