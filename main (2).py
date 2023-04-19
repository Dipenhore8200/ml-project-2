import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
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

# Load the pre-trained model
model = model = load_ckpt("res18_10 .pth")

# Define the class labels
class_labels = ['class1', 'class2', 'class3', ...]

# Create a streamlit app
st.title("Image Classification App")

# Allow the user to upload an image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Preprocess the image
    img = Image.open(uploaded_image)
    

    log_ps = mlpredict(img, model)
    cls_score = int(torch.argmax(torch.exp(log_ps)))

    # Use the pre-trained model to classify the image


    # Display the classification result to the user
    if(cls_score == 0):                                                      
        st.write("Predicted class:", "ELIGATOR")
    if(cls_score == 1):                                                      
        st.write("Predicted class:", "CROCODILE")
    if(cls_score == 2):                                                      
        st.write("Predicted class:", "GHARIAL")
if __name__=='__main__':
    main()
