import tflearn
from model import buildModel

model = buildModel()
images, labels, testImages, testLabels = tflearn.datasets.mnist.load_data(one_hot=True)
model.fit(images, labels, n_epoch=20, validation_set=(testImages, testLabels), show_metric=True)
model.save("mnist.model")