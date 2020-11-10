import os
from distutils.dir_util import copy_tree

path = "/ws/data"
class_path = os.path.join(path, "imagenet-a/README.txt")

with open(class_path, 'r') as f:
    data = f.readlines()
    class_list = data[12:]

for row in class_list:
    class_name = row.split(" ")[0]
    train_path = "/ws/data/imagenet/train/" + class_name
    new_train_path = "/ws/data/imagenet200/train/" + class_name
    copy_tree(train_path, new_train_path)
    # val_path = "/ws/data/imagenet/val/" + class_name
    # new_val_path = "/ws/data/imagenet200/val/" + class_name
    # copy_tree(val_path, new_val_path)

    # if not os.path.exists(new_train_path):
    #     os.mkdir(new_train_path)
    # if not os.path.exits(new_val_path):
    #     os.mkdir(new_val_path)
