import tflearn


def buildModel():
    inputLayer = tflearn.input_data(shape=[None, 784])
    l1 = tflearn.fully_connected(inputLayer, 64, activation='tanh', regularizer='L2')
    drop1 = tflearn.dropout(l1, 0.8)
    l2 = tflearn.fully_connected(drop1, 64, activation='tanh', regularizer='L2')
    drop2 = tflearn.dropout(l2, 0.8)
    output = tflearn.fully_connected(drop2, 10, activation='softmax')

    net = tflearn.regression(output,
                             optimizer=tflearn.SGD(learning_rate=0.1),
                             metric=tflearn.metrics.Accuracy(),
                             loss='categorical_crossentropy')
    return tflearn.DNN(net)
