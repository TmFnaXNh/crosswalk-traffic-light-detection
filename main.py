import torch
import yaml
import argparse
import os
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ExifTags
from cnn import TrafficLightCNN
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import pandas as pd
import os
def find_next_available_dir(base_dir):
    i = 1
    while True:
        new_dir = os.path.join(base_dir, f'exp{i}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1
def load_and_transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)
def predict_image(model, image_tensor, device):
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()
def fix_image_orientation(image_path):
    image = Image.open(image_path)
    try:
        exif = image._getexif()
        orientation_key = next(key for key, val in ExifTags.TAGS.items() if val == 'Orientation')
        if exif[orientation_key] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation_key] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation_key] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    return image
def crop_image(image, x1, y1, x2, y2):
    image_width, image_height = image.size
    new_x1 = max(0, x1 - (x2 - x1) * 0.5)
    new_y1 = 0
    new_x2 = min(image_width, x2 + (x2 - x1) * 0.5)
    new_y2 = min(image_height, y1 * 1.2)
    cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))
    return cropped_image
cnt = 0
def save_image(result, image_path):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    img = result.render()[0]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, img_bgr)
def process_image(image_path, crosswalk_model, signal_model, cnn_weights_model, root):
    image = fix_image_orientation(image_path)
    signal_results = signal_model(image_path)
    crossing_results = crosswalk_model(image_path)
    save_path = f'{root}/{image_path[image_path.rfind("/") + 1:image_path.rfind(".")]}'
    save_image(signal_results, save_path + '/signal.png')
    save_image(crossing_results, save_path + '/crossing.png')
    crossing_upmid = image.size[0] / 2, image.size[1]
    if len(signal_results.xyxy[0]) > 1 or len(signal_results.xyxy[0]) == 0:
        if len(crossing_results.xyxy[0]) > 0:
            cropped_image = crop_image(image, float(crossing_results.xyxy[0][0][0]), float(crossing_results.xyxy[0][0][1]), float(crossing_results.xyxy[0][0][2]), float(crossing_results.xyxy[0][0][3]))
            cropped_signal_results = signal_model(cropped_image)
            if len(cropped_signal_results.xyxy[0]) > 0:
                signal_results = cropped_signal_results
                image = cropped_image
                crossing_upmid = image.size[0] / 2, image.size[1]
    if len(signal_results.xyxy[0]) == 0:
        return None
    score_m, img_info = 0, None
    max_area = max([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2, conf, class_typ in signal_results.xyxy[0]])
    max_dist = max([((x1 + x2) / 2 - crossing_upmid[0]) ** 2 + ((y1 + y2) / 2 - crossing_upmid[1]) ** 2 for x1, y1, x2, y2, conf, class_typ in signal_results.xyxy[0]])
    for x1, y1, x2, y2, conf, class_typ in signal_results.xyxy[0]:
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        dist = (crossing_upmid[0] - x_center) ** 2 + (crossing_upmid[1] - y_center) ** 2
        area = (x2 - x1) * (y2 - y1)
        dist /= max_dist
        area /= max_area
        if score_m < area * 0.5 + (1 - dist) * 0.5:
            score_m = area * 0.5 + (1 - dist) * 0.5
            img_info = float(x1), float(y1), float(x2), float(y2)
    final_image = image.crop(img_info)
    image_tensor = load_and_transform_image(final_image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_class_index = predict_image(cnn_weights_model, image_tensor, device)
    with open(save_path + '/class.txt', 'w') as f:
        if predicted_class_index == 0:
            f.write('crosswalk signal state: red\n')
        else:
            f.write('crosswalk signal state: green\n')
        f.write('detected signals: ' + str(len(signal_results.xyxy[0])) + '\n')
        f.write('detected crossings: ' + str(len(crossing_results.xyxy[0])) + '\n')
def main(crosswalk_weights_file, signal_weights_file, img_dir, cnn_weights_file, root):
    cnn_weights_model = TrafficLightCNN()
    cnn_weights_model.load_state_dict(torch.load(cnn_weights_file))
    # change settings in ~/.cache/torch/hub/ultralytics_yolov5_master
    crosswalk_model = torch.hub.load('ultralytics/yolov5', 'custom', path=crosswalk_weights_file)
    crosswalk_model.hyp = 'crosswalk'
    signal_model = torch.hub.load('ultralytics/yolov5', 'custom', path=signal_weights_file)
    signal_model.hyp = 'traffic light'
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        process_image(img_path, crosswalk_model, signal_model, cnn_weights_model, root)
if __name__ == "__main__":
    crosswalk_weights = './models/crossing(cbam)/weights/best.pt'
    signal_weights = './models/signal(se)/weights/best.pt'
    image_dir = './data/traffic lights/test/images'
    cnn_weights = './models/cnn/best_cnn.pth'
    root = find_next_available_dir('./runs')
    main(crosswalk_weights, signal_weights, image_dir, cnn_weights, root)
