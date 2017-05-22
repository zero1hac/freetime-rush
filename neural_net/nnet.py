from numpy import exp, array, random, dot

class NeuralNet():
    def __init__(self):
        random.seed(1)

        # Assigning random weights to the network
        self.synapse = 2*random.random((3,1)) - 1


    def __activation(self, x):
        return 1 / (1 + exp(-x))


    def __activation_derivative(self, x):
        return x * (1 - x)


    def train(self, train_x, train_y, iter):
        for i in range(iter):
            y1 = self.predict(train_x)

            err = train_y - y1
            update = dot(train_x.T, err * self.__activation_derivative(train_y))

            self.synapse += update


    def predict(self, inputs):
        return self.__activation(dot(inputs, self.synapse))

if __name__ == "__main__":
    nn = NeuralNet()

    print "Initializing weights"
    print nn.synapse

    train_x = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_y = array([[0, 1, 1, 0]]).T

    nn.train(train_x, train_y, 100000)


    print "Final synapses"
    print nn.synapse

    print "predict the output value for [1, 0, 0]"
    print nn.predict(array([[1, 0, 0]]))

    print "predict the output value for [1, 1, 1]"
    print nn.predict(array([[1, 1, 1]]))
