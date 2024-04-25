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

# NEW IMPORTS
import easyocr
from shapely.geometry import Polygon
from shapely.ops import unary_union
import json
from manga_ocr import MangaOcr
from googletrans import Translator


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
@app.route('/textbox_dummy', methods=['POST'])
def gen_textboxes_dummy():
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
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco"
        },
        {
            "top": 100,
            "left": 150,
            "right": 550,
            "bottom": 125,
            "text": "Another sample of extracted text, possibly from a different part of the same document or image."
        },
        {
            "top": 300,
            "left": 100,
            "right": 600,
            "bottom": 350,
            "text": "Here is some more text, extracted from an area below the previous boxes, showcasing multiline capability."
        },
        {
            "top": 400,
            "left": 120,
            "right": 580,
            "bottom": 420,
            "text": "Final example of text recognition box, smaller and in the lower section of the image."
        }
    ]

    return jsonify(response)


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
    
    # Save uploaded files temporarily
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
    image.save(image_path)

    # RUN EASY OCR 
    reader = easyocr.Reader(['ja','en']) # this needs to run only once to load the model into memory
    result = reader.readtext(image_path)
    bounding_boxes = [detection[0] for detection in result]

    # MERGE BOUNDING BOXES
    merged_boxes = merge_boxes(bounding_boxes)
    merged_boxes = merge_boxes(merged_boxes, dist_threshold=0)

    print('----Merged Boxes------------------------------------------------')
    print("merged_boxes", merged_boxes)
    print('----------------------------------------------------')

    translations = run_magaOCR(image_path, merged_boxes)

    print('----Translations------------------------------------------------')
    print("translations", translations)
    print('----------------------------------------------------')

    response = translation_json(translations)


    return response


# helpers for bounding box
def get_minimum_bounding_box(polygons):
    """
    Get the minimum bounding box for a list of polygons.
    """
    merged_poly = unary_union(polygons)  # Merge all polygons into one
    min_rect = merged_poly.minimum_rotated_rectangle  # Get the minimum rotated rectangle
    return np.array(min_rect.exterior.coords)
# helpers for bounding box
def merge_boxes(bounding_boxes, dist_threshold=10):
    """
    Merge rotated bounding boxes that are overlapping or within a specified distance threshold.
    """
    polygons = [Polygon(box) for box in bounding_boxes]
    
    # Merge polygons that are close to each other or overlap
    merged_polygons = []
    for poly in polygons:
        if not merged_polygons:
            merged_polygons.append(poly)
            continue
        
        for merged_poly in merged_polygons:
            if poly.distance(merged_poly) < dist_threshold or poly.intersects(merged_poly):
                merged_polygons.remove(merged_poly)
                poly = unary_union([poly, merged_poly])
                break
        merged_polygons.append(poly)
    
    # Get the minimum bounding boxes for the merged polygons
    merged_boxes = [get_minimum_bounding_box([poly]) for poly in merged_polygons]
    
    return merged_boxes

# USE MANGA-OCR TO GET JAPANESE CHARACTERS
def run_magaOCR(image_path, merged_boxes):
    """ GENERATE ARRAY OF TRANSLATIONS """
    print('~~~~~~~~~~ !!!!!!!!!!!!!!!!! ATTEMPTING MangaOCR INIT !!!!!!!!!!!!!!!!!!!! ~~~~~~~~~~~')
    mocr = MangaOcr()
    print('~~~~~~~~~~ MOCR COMPLETE ~~~~~~~~~~~')

    img = cv.imread(image_path)

    translations = []

    def get_rectangle_coords(box):
        """Convert bounding box coordinates to rectangle coordinates."""
        x_min = np.min(box[:, 0])
        y_min = np.min(box[:, 1])
        x_max = np.max(box[:, 0])
        y_max = np.max(box[:, 1])
        return int(x_min), int(y_min), int(x_max), int(y_max)

    for box in merged_boxes:
        x_min, y_min, x_max, y_max = get_rectangle_coords(box)
        cropped_image = img[y_min:y_max, x_min:x_max]
        
        cropped_image_rgb = cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB) # convert the cropped image from OpenCV's BGR format to RGB
        cropped_image_pil = PIL.Image.fromarray(cropped_image_rgb) # convert the NumPy array to a PIL.Image object
        
        text = mocr(cropped_image_pil)  # call api
        
        translations.append([box,text])
    
    return translations

def translation_json(input_data):
    result = []
    for item in input_data:
        coordinates = item[0]
        japanese_text = item[1]
        english_text = translate_japanese_to_english(japanese_text)
        xs = [coord[0] for coord in coordinates]
        ys = [coord[1] for coord in coordinates]
        bounding_box = {
            "top": max(ys),
            "left": min(xs),
            "right": max(xs),
            "bottom": min(ys),
            "text": english_text
        }
        result.append(bounding_box)
    return json.dumps(result, indent=4, ensure_ascii=False)

def translate_japanese_to_english(text):
    translator = Translator()
    translated_text = translator.translate(text, src='ja', dest='en')
    return translated_text.text