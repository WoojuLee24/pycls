import os

path = "/ws/external/checkpoints/c1000/baseline_3gpu/imagenet-c_stdout.log"
new_path = "/ws/external/checkpoints/c1000/baseline_3gpu/imagenet-c_stdout_revised.log"
data_path = "/ws/data/imagenet-c"

with open(path, 'r') as f:
    rows = f.readlines()

errs = []
corruptions = []

for folder in sorted(os.listdir(data_path)):
    sub_path = os.path.join(data_path, folder)
    for sub_folder in sorted(os.listdir(sub_path)):
        sub_path2 = os.path.join(sub_path, sub_folder)
        corruptions.append(sub_folder)
        with open(new_path, 'a') as f:
            f.write(sub_folder + ", ")

errs= []
k = 0
for row in rows:
    if "test_epoch" in row:
        row_elem = row.split(":")
        min_top1_err = row_elem[7].split(",")[0]
        min_top5_err = row_elem[8].split(",")[0]
        # print("catetory: {}, corruption_class: {}, level: {}, min_top1_err: {}".format(catetory, corruption_class, level, min_top1_err))
        if k % 5 == 0:
            errs.append(corruptions[k//5])
        errs.append(float(min_top1_err)/100)
        # errs.append(f"{float(min_top1_err) / 100:0.4f}")
        k+=1

with open(new_path, 'a') as f:
    f.write(str(errs)+"\n")
