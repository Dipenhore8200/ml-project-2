from flask import Flask, request, render_template
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models
import numpy as np
import PIL
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an torch Tensor
    '''
    im = PIL.Image.open(image)
    return test_transforms(im)


def mlpredict(image_path, model):
    # Predict the class of an image using a trained deep learning model.
    model.eval()
    img_pros = process_image(image_path)
    img_pros = img_pros.view(1,3,224,224)
    with torch.no_grad():
        output = model(img_pros)
    return output
# Load saved model
def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(512, 400)),
                      ('relu', nn.ReLU()),
                      ('fc2', nn.Linear(400, 3)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))

    model.load_state_dict(ckpt, strict=False)

    return model

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
##img_path = "C:\\Users\\Dipen\\Downloads\\th.jpeg"



app = Flask(__name__)
model = load_ckpt("res18_10 .pth") # Replace with your pre-trained model

@app.route("/")
def home():
    return render_template("ml webpage.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    img = request.files["image"]
    log_ps = mlpredict(img, model)
    cls_score = int(torch.argmax(torch.exp(log_ps)))
    
    if(cls_score == 0):
        return render_template("predict.html", prediction="elegator")
    elif(cls_score == 1):
        return render_template("predict.html", prediction="crocodile")
    elif(cls_score == 2):
        return render_template("predict.html", prediction="gharial")

if __name__ == "__main__":
    app.run(debug=True)
