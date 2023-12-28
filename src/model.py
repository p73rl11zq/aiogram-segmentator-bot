import numpy as np
import os
import shutil
import torch
from torchvision import transforms
from pathlib import Path
import onnx
import onnxruntime
from PIL import Image
import cv2
import gdown

from src.process import *

class Model():
    def __init__(self):
        self.model_path = os.getcwd()+'/bisenet.onnx'
        if os.path.exists(self.model_path):
            pass
        else:
            url = 'https://drive.google.com/uc?id=1k5RnJNg6boQVT_qXdLH0Z2O3Gr6XdV0S'
            gdown.download(url, self.model_path, quiet=False)
        
        self.instance = onnxruntime.InferenceSession(self.model_path)

    def predict(self, input_path, save_name=None, save_im=True):
        image = cv2.imread(input_path)
        original_shape = image.shape[:2][::-1] # cv 'feature' is to be getting destroyed
        image = cv2.resize(image, (512,512))
        frame = preprocess(image)
        input_dict = {"input_0": frame}
        pred = self.instance.run(None, input_feed = input_dict)
        ort_outs_pt = torch.Tensor(pred[0])
        parsing = ort_outs_pt.squeeze(0).cpu().numpy().argmax(0)
        img2save = vis_parsing_maps(image, parsing, stride=1)

        if save_im:
            if save_name is None:
                save_name = os.getcwd()+'/mask.png'
            else:
                pass
            cv2.imwrite(save_name, cv2.resize(img2save, original_shape), [int(cv2.IMWRITE_JPEG_QUALITY), 100])