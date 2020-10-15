import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualization(path, datas):

    for i in range(len(datas)):
        img_name = datas[i]["img_name"]
        img = datas[i]["img"]
        ano = datas[i]["ano"]
        output = datas[i]["output"]

        img = img.permute(1,2,0).to('cpu').detach().numpy().copy()
        ano = ano.to('cpu').detach().numpy().copy()
        output = output.to('cpu').detach().numpy().copy()

        fig = plt.figure()
        plt.imshow(np.squeeze(img))
        plt.colorbar()
        fig.savefig(os.path.join(path, img_name + "_img.png"))
        plt.close()

        fig = plt.figure()
        plt.imshow(np.squeeze(ano))
        plt.colorbar()
        fig.savefig(os.path.join(path, img_name + "_ano.png"))
        plt.close()

        fig = plt.figure()
        plt.imshow(np.squeeze(output))
        plt.colorbar()
        fig.savefig(os.path.join(path, img_name + "_output.png"))
        plt.close()
