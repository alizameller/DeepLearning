#!/bin/env python
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../hw1")
from hw1 import Linear, grad_update

class Siren(Linear, tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation,
        output_activation,
    ):
        self.M = hidden_layer_width
        self.K = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.input_linear = Linear(num_inputs, self.M)
        self.hidden_linear = [Linear(self.M, self.M) for i in range(self.K)]
        self.output_linear = Linear(self.M, num_outputs)

    def __call__(self, x):
        p = self.input_linear(x)
        for i in range(self.K):
            p = self.hidden_activation(p)
            p = self.hidden_linear[i](p)

        p = self.output_linear(p)
        return self.output_activation(p)

def train(model, model_optimizer, nb_epochs=15000):
    psnr = []
    for _ in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)
        loss = ((model_output - pixel_values) ** 2).mean()
        psnr.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    return psnr, model_output

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt
    import yaml
    from tqdm import trange
    import cv2

    hidden_layer_width = 256
    num_hidden_layers = 4
    num_inputs = 2
    num_outputs = 1
    num_samples = 500

    siren = Siren(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.math.sin,
        output_activation=tf.math.sin,
    )

    img = cv2.imread('Testcard_f.jpg')
    print(img.shape)

    # Target
    #img = ((torch.from_numpy(skimage.data.camera()) - 127.5) / 127.5)
    pixel_values = img.reshape(-1, 1)

    # Input
    resolution = img.shape[0]
    tmp = tf.linspace(-1, 1, resolution)
    x, y = tf.meshgrid(tmp, tmp)
    # pixel_coordinates = tf.concat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=13)
    # plt.show()
    breakpoint()
    
    for i in enumerate([siren]):
        # Training
        # optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
        # psnr, model_output = train(model, optim, nb_epochs=15000)
        axes[i + 1].imshow(model_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
        #axes[i + 1].set_title('ReLU' if (i == 0) else 'SIREN', fontsize=13)
        #axes[4].plot(psnr, label='ReLU' if (i == 0) else 'SIREN', c='C0' if (i == 0) else 'mediumseagreen')
        #axes[4].set_xlabel('Iterations', fontsize=14)
        #axes[4].set_ylabel('PSNR', fontsize=14)
        #axes[4].legend(fontsize=13)

    for i in range(4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[3].axis('off')
    plt.savefig('Imgs/Siren.png')
    plt.close() 
    '''
