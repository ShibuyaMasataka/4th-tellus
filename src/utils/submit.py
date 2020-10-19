import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

def save_json(path, datas):

    submit = {}
    for i in range(len(datas)):
        img_name = datas[i]["img_name"] + '.tif'
        points = datas[i]["points"]
        points = points.to('cpu').detach().numpy().copy()

        submit[img_name] = points.astype(np.int).tolist()
    with open(os.path.join(path, "submit.json"), 'w') as f:
        json.dump(submit, f)
