import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="/ws/external", type=str)

args = parser.parse_args()
path = os.path.join(args.path, "imagenet-c/stdout.log")
new_path = os.path.join(args.path, "imagenet-c/stdout_revised.log")

# path = "/ws/external/checkpoints/c200/c200_End_bottleneck_dilation_fc_entire/imagenet-c/stdout.log"
# new_path = "/ws/external/checkpoints/c200/c200_End_bottleneck_dilation_fc_entire/imagenet-c/stdout_revised.log"

with open(path, 'r') as f:
    rows = f.readlines()

errs = []
corruptions = ["gaussian_noise.npy", "shot_noise.npy", "impulse_noise.npy",
               "defocus_blur.npy", "glass_blur.npy", "motion_blur.npy", "zoom_blur.npy",
               "snow.npy", "frost.npy", "fog.npy", "brightness.npy",
               "contrast.npy", "elastic_transform.npy", "pixelate.npy", "jpeg_compression.npy",
               "speckle_noise.npy", "gaussian_blur.npy", "spatter.npy", "saturate.npy"]
errs= []
k = 0
for row in rows:
    if "test_epoch" in row:
        row_elem = row.split(":")
        min_top1_err = row_elem[7].split(",")[0]
        min_top5_err = row_elem[8].split(",")[0]
        # print("catetory: {}, corruption_class: {}, level: {}, min_top1_err: {}".format(catetory, corruption_class, level, min_top1_err))
        errs.append(corruptions[k])
        errs.append(float(min_top1_err)/100)
        # errs.append(f"{float(min_top1_err) / 100:0.4f}")
        k += 1

with open(new_path, 'a') as f:
    f.write(str(errs)+"\n")
