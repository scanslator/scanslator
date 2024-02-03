from flask import Flask, request, send_file
import os
import numpy as np
import PIL.Image
import torch
import torch.backends
import torchvision.transforms.v2
import fastai.vision.learner
import fastai.vision.models
import fastai.vision.data


print('Initializing Model...')
device = torch.device('cpu')
model = fastai.vision.learner.create_unet_model(fastai.vision.models.resnet34, 1, (1656, 1176))
model = model.to(device)
next(model.parameters()).device
model.load_state_dict(torch.load('juvian_weights_fastai2.7.pt'))

print('Initializing Pipeline...')
preprocess_pipeline = torchvision.transforms.v2.Compose([
    torchvision.transforms.v2.ToImage(),
    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
    torchvision.transforms.v2.Normalize(*fastai.vision.data.imagenet_stats),
])
def preprocess(fn):
    return preprocess_pipeline(PIL.Image.open(fn)).unsqueeze(0)

def infer(xb):
    with torch.no_grad():
        return model(xb)
    
def postprocess(logits):
    # analyze_pred
    y = (torch.sigmoid(logits.squeeze()) > 0.5).type(torch.uint8) * 255
    img = PIL.Image.fromarray(y.numpy(), mode='L')
    return img

def postprocess_red(logits):
    thresholded = (torch.sigmoid(logits.squeeze()) > 0.5).numpy()
    rgba_image = np.zeros((thresholded.shape[0], thresholded.shape[1], 4), dtype=np.uint8)
    rgba_image[thresholded == 1] = [255, 0, 0, 255]    
    img = PIL.Image.fromarray(rgba_image, mode='RGBA')
    return img

print('Initializing API...')
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    return "<p>Hello, Scanslator!</p>"

@app.route('/mask', methods=['POST'])
def gen_mask():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            test_input = preprocess(file).to(device)
            test_logits = infer(test_input).cpu()
            mask = postprocess_red(test_logits)
            processed_filename = "processed_" + os.path.splitext(file.filename)[0] + ".png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            mask.save(filepath, format='PNG')
            return send_file(filepath, mimetype='image/png')
    return