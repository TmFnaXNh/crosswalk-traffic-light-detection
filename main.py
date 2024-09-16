"""
copy code in "./yolov5-master/models/common.py" to your "Users/username/.cache/torch/hub/ultralytics_yolov5_master/models/common.py" to run this code.
download our dataset from "https://drive.google.com/drive/folders/1E3qQOaw82gZIkMg-UFsIMB_4cs0jCLDX?usp=drive_link" and put the downloaded data folder in the same directory.
"""

import torch
import yaml
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ExifTags
from cnn import TrafficLightCNN
def load_and_transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = image.convert('RGB')
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
    new_x1 = max(0, x1 - (x2 - x1) * 0.3)
    new_y1 = 0
    new_x2 = min(image_width, x2 + (x2 - x1) * 0.3)
    new_y2 = min(image_height, y1 * 1.2)
    cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))
    return cropped_image
cnt = 0
def process_image(image_path, crosswalk_model, signal_model, cnn_weights_file):
    image = fix_image_orientation(image_path)
    signal_results = signal_model(image)
    crossing_upmid = image.size[0] / 2, image.size[1]
    if len(signal_results.xyxy[0]) > 1 or len(signal_results.xyxy[0]) == 0:
        crossing_results = crosswalk_model(image_path)
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
    model = TrafficLightCNN()
    model.load_state_dict(torch.load(cnn_weights_file))
    image_tensor = load_and_transform_image(final_image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_class_index = predict_image(model, image_tensor, device)
    return image_path, predicted_class_index
def main(crosswalk_weights_file, signal_weights_file, img_dir, label_dir, cnn_weights_file):
    cnt1, cnt2, cnt3 = 0, 0, 0
    valid_prefix = [file[:-4] for file in os.listdir(label_dir)]
    crosswalk_model = torch.hub.load('ultralytics/yolov5', 'custom', path=crosswalk_weights_file)
    crosswalk_model.hyp = 'crosswalk'
    signal_model = torch.hub.load('ultralytics/yolov5', 'custom', path=signal_weights_file)
    signal_model.hyp = 'traffic light'
    result = []
    for filename in os.listdir(img_dir):
        if filename[:filename.index('.')] not in valid_prefix:
            continue
        img_path = os.path.join(img_dir, filename)
        result.append(process_image(img_path, crosswalk_model, signal_model, cnn_weights_file))
        if result[-1] == None:
            cnt2 += 1
            continue
        with open(os.path.join(label_dir, filename[:-4] + '.txt'), 'r') as f:
            lines = f.readlines()
            if (int(result[-1][1]) == 1 and lines[0][0] == '1') or (int(result[-1][1]) == 0 and lines[0][0] == '0'):
                cnt3 += 1
            else:
                cnt1 += 1
        print(f"Name:{filename}, Predicted:{result[-1][1] ^ 1}, Ground Truth:{lines[0][0]}")
    print(f"Right:{cnt1}, Not Detected:{cnt2}, Wrong:{cnt3}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic signals Detector")
    parser.add_argument('--crosswalk_weights', type=str, help="Path to the crosswalk YOLOv5 weights file", default='./models/crossing(cbam)/weights/best.pt')
    parser.add_argument('--signal_weights', type=str, help="Path to the signal YOLOv5 weights file", default='./models/signal(se)/weights/best.pt')
    parser.add_argument('--image_dir', type=str, help="Path to the directory containing image files for detection", default='./data/images(all)')
    parser.add_argument('--label_dir', type=str, help="Path to the directory containing label files for detection", default='./data/TFlabels(crosswalk traffic light)')
    parser.add_argument('--cnn_weights', type=str, help="Path to the CNN weights file", default='./models/cnn/best_cnn.pth')
    args = parser.parse_args()
    main(args.crosswalk_weights, args.signal_weights, args.image_dir, args.label_dir, args.cnn_weights)