import os

path = "/ws/external/imagenet-c_stdout.log"
new_path = "/ws/external/imagenet-c_revised_stdout.log"

with open(path, 'r') as f:
    rows = f.readlines()

errs = []
corruptions = []
for row in rows:
    if "data path" in row:
        row_elem = row.split("/")
        catetory = row_elem[4]
        corruption_class = row_elem[5]
        level = row_elem[6]
    elif "test_epoch" in row:
        row_elem = row.split(":")
        min_top1_err = row_elem[7].split(",")[0]
        min_top5_err = row_elem[8].split(",")[0]
        # print("catetory: {}, corruption_class: {}, level: {}, min_top1_err: {}".format(catetory, corruption_class, level, min_top1_err))
        errs.append(float(min_top1_err))
        corruptions.append(corruption_class)
        with open(new_path, 'a') as f:
            f.write("catetory: {}, corruption_class: {}, level: {}, min_top1_err: {}\n".format(catetory, corruption_class, level, min_top1_err))

with open(new_path, 'a') as f:
    f.write(str(errs)+"\n")
    f.write(str(corruptions))
