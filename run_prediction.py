import subprocess
import glob
import os


source = "Data/Test_DS/images"


weights = glob.glob("Data/cp/*")

for weight in weights:

    cmd = f"yolo predict model={weight} iou=0.01 conf=0.05 source={source} save_txt save_conf project=Data/preds/test_preds name={source.split('/')[-2]}_{os.path.basename(weight).split('.')[0]}"
    print(f"* Command:\n{cmd}")
    subprocess.call(cmd, shell=True)
