import neural.NeuralNetwork as NeuralNetwork
import neural.loss as loss

def training(num_epochs, data, g_truth, training_rate = 0.01):
    nn = NeuralNetwork()
    losses = []
    for i in range(num_epochs):
        # create loss
        loss_list = [loss(nn.forward(data_i), gt_i) for data_i, gt_i in zip(data, g_truth)]
        l = sum(loss_list) * (1.0/len(loss_list))
        losses.append(l.value)
        
        # generate gradients
        nn.zero_grad()
        l.backward()

        #update gradients
        for p in nn.params():
            p.value -= training_rate * p.grad
        print(f"loss: {l.value}")
    
