from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import numpy as np
import PIL.Image
import torch
import torch.backends
import torchvision.transforms.v2
import fastai.vision.learner
import fastai.vision.models
import fastai.vision.data
import cv2 as cv
from werkzeug.utils import secure_filename


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

def infill(image_path, mask_path):
    img = cv.imread(image_path)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    
    # Dilate the mask if needed (currently commented out)
    kernel = np.ones((10,10), np.uint8)
    dilated_mask = cv.dilate(mask, kernel, iterations = 1)
    
    # Apply inpainting with mask
    dst = cv.inpaint(img, dilated_mask, 10, cv.INPAINT_TELEA)
    return dst
    

print('Initializing API...')
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# test route
@app.route("/")
def hello_world():
    return "<p>Hello, Scanslator!</p>"

# generate mask
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


# generate infill
@app.route('/infill', methods=['POST'])
def gen_infill():
    if request.method == 'POST':
        if 'mask' not in request.files or 'image' not in request.files:
            return 'No file part in the request', 400
        image = request.files['image']
        mask = request.files['mask']
        if image.filename == '':
            return 'No selected file', 400
        if image and mask:
            # Save uploaded files temporarily
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(mask.filename))
            image.save(image_path)
            mask.save(mask_path)
            
            # Process the image and mask
            output = infill(image_path, mask_path)
            processed_filename = "infill_" + os.path.splitext(image.filename)[0] + ".png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            # Save the processed image
            cv.imwrite(filepath, output)
            
            # Return the processed file
            return send_file(filepath, mimetype='image/png')
    return 'Invalid request', 400

# generate translation with textbox
@app.route('/textbox', methods=['POST'])
def gen_textboxes():
    if request.method != 'POST':
        return 'Invalid request method', 405

    if 'image' not in request.files:
        return 'Image file part missing', 400

    image = request.files['image']

    if image.filename == '':
        return 'No image selected', 400

    # Here, you would normally process the image to detect text boxes
    # For now, we just return a fixed JSON response

    response = [
        {
            "top": 189,
            "left": 231,
            "right": 249,
            "bottom": 133,
            "text": "Sample text that has been translated from manga!"
        },
        {
            "top": 777,
            "left": 249,
            "right": 555,
            "bottom": 321,
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco "
        }
    ]

    return jsonify(response)
