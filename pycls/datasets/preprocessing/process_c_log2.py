import os

path = "/ws/external/imagenet-c_stdout.log"
new_path = "/ws/external/imagenet-c_revised_stdout.log"
data_path = "/ws/data/imagenet-c"

with open(path, 'r') as f:
    rows = f.readlines()

errs = []
corruptions = []

for folder in sorted(os.listdir(data_path)):
    sub_path = os.path.join(data_path, folder)
    for sub_folder in sorted(os.listdir(sub_path)):
        sub_path2 = os.path.join(sub_path, sub_folder)
        with open(new_path, 'a') as f:
            f.write(sub_folder+ ", ")

errs=[]
for row in rows:
    if "test_epoch" in row:
        row_elem = row.split(":")
        min_top1_err = row_elem[7].split(",")[0]
        min_top5_err = row_elem[8].split(",")[0]
        # print("catetory: {}, corruption_class: {}, level: {}, min_top1_err: {}".format(catetory, corruption_class, level, min_top1_err))
        errs.append(float(min_top1_err))

with open(new_path, 'a') as f:
    f.write(str(errs)+"\n")
