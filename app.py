from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

classes = ['Pollu_Disease','black_pepper_healthy','black_pepper_leaf_blight']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = models.mobilenet_v3_small()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)

model.load_state_dict(torch.load("pepper_disease_model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET','POST'])
def index():

    prediction = None

    if request.method == 'POST':

        file = request.files['image']
        img = Image.open(file)

        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs,1)

        prediction = classes[predicted.item()]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)