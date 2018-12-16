from artificialneuralnetwoks.artificial_neural_network import AritificalNeuralNetwork

def main():
    a = AritificalNeuralNetwork(5, 3, [3,2,3], 1)
    predictions = a.forward_propagate(a.generate_random_inputs());
    print('The predicted values by the network for the given input are {}'.format(predictions))

if __name__ == '__main__':
    main()
