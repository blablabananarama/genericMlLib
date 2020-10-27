import numpy as np

# define a numpy forward pass
def neural_net():
    ''' creates simple neural net '''
    first_layer = connected_layer(2, 3)
    second_layer = connected_layer(3, 3)
    output_layer = connected_layer(3, 2)
    return [first_layer, second_layer, output_layer]


