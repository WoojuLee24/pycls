from PIL import Image
import numpy as np
import torch
import json

from pycls.core.trainer import setup_model
from pycls.core.config import cfg
import pycls.core.config as config
import pycls.core.checkpoint as cp
from pycls.visualization.misc_functions import get_example, save_class_activation_images

import matplotlib.pyplot as plt
import math
import os


def visualize_weight(model, target_layer, path=None, max_num=100, show=None):

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
    Tot = N * C if N * C < max_num else max_num     # inchannel, outchannel

    cols = int(math.sqrt(Tot))
    rows = Tot // cols + 1
    pos = range(1, Tot + 1)

    fig = plt.figure()
    fig.tight_layout()
    k = 1
    for i in range(cols):
        for j in range(rows):
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
    target_class = "30"  # black swan 100, bullfrog 30, centipede 79, thunder snake 52
    label_path = "/ws/data/imagenet/imagenet_class_index.json"
    target_layer = "stem.e.weight"  # "stem.conv" "s4.b3.f.b"

    # load the model
    config.load_cfg_fom_args("Test a trained classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    pretrained_model = setup_model()
    cp.load_checkpoint(cfg.TEST.WEIGHTS, pretrained_model)

    (original_image, prep_img, class_name, jpg) =\
        get_example(target_class, label_path, target_layer)

    original_path = "/ws/external/visualization_results/filter/" + class_name + "_" + jpg
    output_path = "/ws/external/visualization_results/filter/" + class_name + "_" + target_layer + "_" + jpg

    visualize_weight(pretrained_model, target_layer, path=output_path, max_num=100)
