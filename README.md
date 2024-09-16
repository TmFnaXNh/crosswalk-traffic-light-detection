# Crosswalk Traffic Light Detection

This repository contains a novel algorithm for detecting crosswalk traffic lights, designed to improve the safety of visually impaired pedestrians. The algorithm incorporates SE (Squeeze-and-Excitation) and CBAM (Convolutional Block Attention Module) models to enhance object detection accuracy in challenging scenarios with multiple traffic lights.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [How to Run](#how-to-run)
- [Arguments](#arguments)
- [References](#references)

---

## Project Overview

The **Crosswalk Traffic Light Detection Algorithm** is designed to aid visually impaired individuals in detecting crosswalk traffic lights while addressing challenges such as multi-light interference and small target detection. The model is based on the YOLOv5 architecture and incorporates advanced attention mechanisms to improve detection accuracy.

[Full paper link](https://github.com/你的仓库路径/文件路径)

### Key Features

1. **YOLOv5 Backbone**:
   - Utilizes YOLOv5 for object detection, known for its balance of speed and accuracy.
   - Efficiently detects small objects like crosswalk traffic lights in real-time.

2. **Squeeze-and-Excitation (SE) Module**:
   - Added after the SPPF block of YOLOv5.
   - Enhances the detection of small targets by reweighting feature channels based on their importance.

3. **Convolutional Block Attention Module (CBAM)**:
   - Integrated before each C3 block in the neck and after the SPPF block in the backbone.
   - Uses both channel and spatial attention to improve crosswalk detection accuracy and resolve multi-light interference.

4. **Crosswalk-Associated Detection**:
   - Detects crosswalk locations and uses this information to select the corresponding crosswalk traffic light.
   - Reduces false positives caused by unrelated vehicle traffic lights in multi-light environments.

5. **Custom Dataset**:
   - Dataset captured from a pedestrian's viewpoint, featuring crosswalk and traffic light images under various conditions.
   - Images were labeled for both traffic light and crosswalk detection tasks.

6. **Color Detection via CNN**:
   - Custom CNN designed to classify the color of the detected crosswalk traffic light (red or green).
   - Ensures accurate identification of whether the pedestrian can cross the street.

## Getting Started

To use this project, follow these instructions.

### Prerequisites

You will need to have the following tools installed on your system:

- Python 3.7+
- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/TmFnaXNh/crosswalk-traffic-light-detection.git
   cd crosswalk-traffic-light-detection
   ```

2. **Create a new Conda environment**:

   ```bash
   conda create --name crosswalk_detection python=3.8
   conda activate crosswalk_detection
   ```

3. **Install project dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the YOLOv5 model** from Ultralytics:

   ```bash
   pip install git+https://github.com/ultralytics/yolov5.git
   ```

5. **Move the custom model files**:
   Copy the `common.py` and `yolo.py` files from this repository to the YOLOv5 folder in the `.cache` directory:

   ```bash
   cp ./common.py ~/.cache/torch/hub/ultralytics_yolov5/common.py
   cp ./yolo.py ~/.cache/torch/hub/ultralytics_yolov5/yolo.py
   ```

6. **Download the dataset**:
   Download the dataset from the following link: [Dataset](https://drive.google.com/drive/folders/1E3qQOaw82gZIkMg-UFsIMB_4cs0jCLDX?usp=drive_link).

   After downloading, move the `data` folder to the root of the project directory.

   Example:

    ```bash
   mv /path_to_downloaded_data/crosswalk_data ./data
    ```

### Dataset Setup

The dataset contains images of crosswalk traffic lights used for training and testing. Ensure the dataset is placed under the `data` directory in the project folder.

---

## How to Run

You can run the `main.py` script with the following required arguments:

```bash
python main.py --crosswalk_weights path_to_crosswalk_yolov5_weights \
               --signal_weights path_to_signal_yolov5_weights \
               --image_dir path_to_image_directory \
               --label_dir path_to_label_directory \
               --cnn_weights path_to_cnn_weights
```
