"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import copy
from PIL import Image
from PIL import ImageFilter

import torch
from torch.optim import Adam, SGD
from torchvision import models
from pycls.core.trainer import setup_model
from pycls.core.config import cfg
import pycls.core.config as config
import pycls.core.checkpoint as cp

def preprocess_image(pil_im, resize_im=False):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = torch.tensor(im_as_ten, requires_grad=True, device="cuda")
    return im_as_var

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.cpu().numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
        cnn_layer = "layer1"
        block_pos = 0
        sub_layer = "conv1"
        filter_pos = 5
    """
    def __init__(self, model, selected_layer, selected_filter):

        self.model = model.cuda()
        self.model.eval()
        self.model_name = model.__class__.__name__

        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0

        self.size = 96
        self.initial_learning_rate = 300
        self.weight_decay = 1e-6
        self.upscaling_factor = 1.2
        self.upscaling_steps = 10
        self.iteration_steps = 50
        self.blur = True

        # Create the folder to export images if not exists
        if not os.path.exists('/ws/external/visualization_results/feature'):
            os.makedirs('/ws/external/visualization_results/feature')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        for n, m in self.model.named_modules():
            if n == str(self.selected_layer):
                m.register_forward_hook(hook_function)
        # self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):

        # Hook the selected layer
        self.hook_layer()

        # Generate a random image
        sz = self.size
        self.created_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))

        for i in range(self.upscaling_steps+1):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=self.initial_learning_rate, weight_decay=self.weight_decay)

            for j in range(self.iteration_steps):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                output = self.model(self.processed_image)

                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                loss = -torch.mean(self.conv_output)
                print('Upscaling:', str(i+1), 'Iteration:', str(j+1), 'Loss:', "{0:.2f}".format(loss.cpu().data.numpy()))
                # Backward
                loss.backward()
                # Update image
                optimizer.step()

            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Upscale the image
            sz = int(sz * self.upscaling_factor)
            self.created_image = Image.fromarray(self.created_image)
            self.created_image = self.created_image.resize((sz, sz), resample=Image.BICUBIC)
            self.created_image = self.created_image.filter(ImageFilter.BoxBlur(radius=1))

            # Save image
            if i % 10 == 0:
                im_path = '/ws/external/visualization_results/feature/' + str(self.selected_layer) + \
                          '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)



if __name__ == '__main__':
    cnn_layer = "stem.e" # "s4.b3.f.b"  # "stem.conv"     # "s1.b2.f.b"
    filter_pos = 6
    # Fully connected layer is not needed
    # pretrained_model = models.vgg16(pretrained=True).features
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    pretrained_model = setup_model()
    cp.load_checkpoint(cfg.TEST.WEIGHTS, pretrained_model)
    #pretrained_model = models.resnet50(pretrained=True).eval()
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()



