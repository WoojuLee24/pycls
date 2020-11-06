from PIL import Image
import numpy as np
import torch
import json

from pycls.core.trainer import setup_model
from pycls.core.config import cfg
import pycls.core.config as config
import pycls.core.checkpoint as cp
from pycls.visualization.misc_functions import get_example, save_class_activation_images, save_image

import matplotlib.pyplot as plt
import math
import os


class FeatureMap():

    def __init__(self, model, target_layer, path=None):
        self.model = model
        self.target_layer = target_layer
        self.path = path
        self.model.eval()
        self.hook_layer()

    def hook_layer(self):
        def forward_hook(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = output

        # Hook the selected layer
        for n, m in self.model.named_modules():
            if n == str(self.target_layer):
                m.register_forward_hook(forward_hook)

    def visualize_feature_map(self, input_image):
        model_output = self.model(input_image)
        feature_maps = self.conv_output.data[0].cpu().numpy()
        feature_maps = feature_maps - feature_maps.min()
        feature_maps = feature_maps / (feature_maps.max() + 0.000000000001)

        for i, feature_map in enumerate(feature_maps):
            save_image(feature_map, self.path.format(i))


def visualize_feature_map(model, target_layer, path=None,  cols=None, show=False):

    """Visualize weight and activation matrices learned
    during the optimization process. Works for any size of kernels.

    Arguments
    =========
    kernels: Weight or activation matrix. Must be a high dimensional
    Numpy array. Tensors will not work.
    path: Path to save the visualizations.
    cols: TODO: Number of columns (doesn't work completely yet.)

    """
    model_state = model.state_dict()
    kernels = model_state[target_layer]

    # if not os.path.exists(path):
    #     os.makedirs(path)
    #
    fpath = path
    kernels_data = kernels.cpu().data.numpy()
    kernels = kernels.cpu().detach().clone().numpy()
    kernels = kernels - kernels.min()
    kernels = kernels / (kernels.max() + 0.000000000001)

    N = kernels.shape[0]
    C = kernels.shape[1]
    Tot = N * C  # inchannel, outchannel

    cols = int(math.sqrt(Tot))
    rows = Tot // cols + 1
    pos = range(1, Tot + 1)

    fig = plt.figure()
    fig.tight_layout()
    k = 1
    for i in range(kernels.shape[0]):
        for j in range(kernels.shape[1]):
            img = kernels[i][j]
            ax = fig.add_subplot(rows, cols, k)
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k += 1

    # if not os.path.exists(path + "t"+ str(m)):
    #     os.makedirs(path + "t" + str(m))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if fpath:
        plt.savefig(fpath, dpi=100)
        plt.close(fig)
    if show:
        plt.show()



if __name__ == '__main__':
    # target layer and label
    target_class = "52"  # black swan 100, bullfrog 30, centipede 79, thunder snake 52
    label_path = "/ws/data/imagenet/imagenet_class_index.json"
    target_layer = "stem.conv"  # "stem.conv" "s4.b3.f.b"

    # load the model
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    pretrained_model = setup_model()
    cp.load_checkpoint(cfg.TEST.WEIGHTS, pretrained_model)

    (original_image, prep_img, class_name, jpg) =\
        get_example(target_class, label_path, target_layer)

    original_path = "/ws/external/visualization_results/feature_map/" + class_name + "_" + jpg
    output_path = "/ws/external/visualization_results/feature_map/" + class_name + "_" + target_layer + "_{}_" + jpg

    save_image(original_image, original_path)
    feature_map = FeatureMap(pretrained_model, target_layer, path=output_path)
    feature_map.visualize_feature_map(prep_img)
    # visualize_feature_map(pretrained_model, target_layer, prep_img, path=file_name_to_export)
