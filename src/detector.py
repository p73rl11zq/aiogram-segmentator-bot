import matplotlib.pyplot as plt
from retinaface import pre_trained_models
import torch
import urllib.request
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Retina():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = pre_trained_models.get_model(model_name='resnet50_2020-07-20', max_size=2048, device=self.device)
        self.model.eval()

    def predict(self, input_path, save_path):
        img = Image.open(input_path)
        init_shape = img.size
        new_shape = (512,512)
        kx, ky = init_shape[0]/new_shape[0], init_shape[1]/new_shape[1]

        resized = np.array(img.resize(new_shape))
        img = np.array(img)

        confidence_threshold = 0.95
        nms_threshold = 0.4

        detected_faces = self.model.predict_jsons(
            resized,
            confidence_threshold,
            nms_threshold
            )

        bboxes = []
        confs = []
        for face in detected_faces:
            bboxes.append(face['bbox'])
            confs.append(face['score'])
        ids = np.argsort(confs)
        
        if -1 in confs:
            print("Sorry, i can't find any segmentable faces in image")
            return 0
        else:
            bbox = bboxes[ids[0]]

            new_y1, new_y2 = int(bbox[1]*ky), int(bbox[3]*ky)
            new_x1, new_x2 = int(bbox[0]*kx), int(bbox[2]*kx)
            margin_y = new_y2 - new_y1
            margin_x = int((new_x2 - new_x1) * 0.7)
            
            if (new_y1 - margin_y) < 0:
                margin_y = new_y1
            if (new_x1 - margin_x) < 0:
                margin_x = new_x1
            if (new_y2 + margin_y) > init_shape[1]:
                margin_y = init_shape[1] - new_y2
            if (new_x2 + margin_x) > init_shape[0]:
                margin_x = init_shape[0] - new_x2

            image = img[
                new_y1-margin_y : new_y2+margin_y,
                new_x1-margin_x : new_x2+margin_x
                ]
            image = Image.fromarray(image)
            image.save(save_path +"detected.jpg") 
