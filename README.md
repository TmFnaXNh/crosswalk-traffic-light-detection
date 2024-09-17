# Crosswalk Traffic Light Detection

This repository contains a novel algorithm for detecting crosswalk traffic lights, designed to improve the safety of visually impaired pedestrians. The algorithm incorporates SE (Squeeze-and-Excitation) and CBAM (Convolutional Block Attention Module) models to enhance object detection accuracy in challenging scenarios with multiple traffic lights.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Overview

The **Crosswalk Traffic Light Detection Algorithm** is designed to aid visually impaired individuals in detecting crosswalk traffic lights while addressing challenges such as multi-light interference and small target detection. The model is based on the YOLOv5 architecture and incorporates advanced attention mechanisms to improve detection accuracy.

### Key Features

1. **YOLOv5 Backbone**:
   - Utilizes YOLOv5 for object detection, known for its balance of speed and accuracy.
   - Efficiently detects small objects like crosswalk traffic lights in real-time.

2. **Squeeze-and-Excitation (SE) Module**:
   - Added after the SPPF block of YOLOv5 model for detecting traffic lights.
   - Enhances the detection of small targets by reweighting feature channels based on their importance.
   <img src="https://cdn.luogu.com.cn/upload/image_hosting/swj2z6ot.png" alt="Crosswalk Detection" width="300" />


3. **Convolutional Block Attention Module (CBAM)**:
   - Integrated before each C3 block in the neck and after the SPPF block in the backbone of YOLOv5 model for crosswalk detection.
   - Uses both channel and spatial attention to improve crosswalk detection accuracy and resolve multi-light interference.
   <img src="https://cdn.luogu.com.cn/upload/image_hosting/yu3pjihf.png" alt="Crosswalk Detection" width="300"  />

4. **Crosswalk-Associated Detection**:
   - Detects crosswalk locations and uses this information to select the corresponding crosswalk traffic light.
   - Reduces false positives caused by unrelated vehicle traffic lights in multi-light environments.

5. **Custom Dataset**:
   - Dataset captured from a pedestrian's viewpoint, featuring crosswalk and traffic light images under various conditions.
   - Images were labeled for both traffic light and crosswalk detection tasks.
   - Some of the data used in this project were sourced from the publicly available dataset at [ImVisible](https://github.com/samuelyu2002/ImVisible), while the rest of the data consists of images that we captured ourselves specifically for the purpose of crosswalk and traffic light detection.

6. **Color Detection via CNN**:
   - Custom CNN designed to classify the color of the detected crosswalk traffic light (red or green).
   - Ensures accurate identification of whether the pedestrian can cross the street.
   <img src="https://cdn.luogu.com.cn/upload/image_hosting/58co05e1.png" alt="Crosswalk Detection" width="300" />

### Model Performance
| Model                          | Precision     | Recall         | mAP@0.5     | mAP@0.5:0.95  |
|--------------------------------|---------------|----------------|-------------|---------------|
| Traffic Light YOLOv5s(SE)      | 0.951         | 0.932          | 0.968       | 0.587         |
| Crosswalk YOLOv5s(CBAM)        | 1.000         | 0.975          | 0.975       | 0.779         |

| Model             | Correct | Not Detected | Incorrect | Precision | Recall  | Accuracy |
|-------------------|---------|--------------|-----------|-----------|---------|----------|
| **Our Algorithm**  | 585     | 5            | 10        | 98.3%     | 99.2%   | 97.5%    |
| **YOLOv5 + Pixel** | 552     | 5            | 43        | 92.8%     | 99.1%   | 92.0%    |

## Getting Started

To use this project, follow these instructions.

### Prerequisites

You will need to have the following tools installed on your system:

- Python 3.8+
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
   conda create --name detection python=3.8
   conda activate detection
   ```

3. **Install project dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
   
4. **Move the custom model files**:
   Copy the `common.py` and `yolo.py` files from this repository to the YOLOv5 folder in the `.cache` directory:

   ```bash
   cp ./common.py ~/.cache/torch/hub/ultralytics_yolov5_master/models/common.py
   cp ./yolo.py ~/.cache/torch/hub/ultralytics_yolov5_master/yolo.py
   ```

5. **Download the dataset**:
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

- `--crosswalk_weights`: Path to the YOLOv5 weights for crosswalk detection.
- `--signal_weights`: Path to the YOLOv5 weights for traffic signal detection.
- `--image_dir`: Directory containing the images to be processed for detection.
- `--label_dir`: Directory of the labels (in `.txt` format, containing 0/1 labels) to check the detection validity.
- `--cnn_weights`: Path to the CNN model weights used for additional classifications.

## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). ImageNet classification with deep convolutional neural networks. *Communications of the ACM*, 60(6), 84-90.
2. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2015). Region-based convolutional networks for accurate object detection and segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 38(1), 142-158.
3. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
4. Lin, T.-Y., Goyal, P., et al. (2020). Focal loss for dense object detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(2), 318-327.
5. Jocher, G. (2020). YOLOv5. Available: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
6. Swami, P. S., & Futane, P. (2018). Traffic light detection system for low vision or visually impaired person through voice. *Fourth International Conference on Computing Communication Control and Automation (ICCUBEA)*, Pune, India.
7. Eto, S., Wada, Y., & Wada, C. (2023). Convolutional neural network based zebra crossing and pedestrian traffic light recognition. *Journal of Mechanical and Electrical Intelligent System*, 6(3), 1-11.
8. Rao, V., & Nguyen, H. (2024). A computer vision based system to make street crossings safer for the visually impaired. *Journal of High School Science*, 8(2), 253-266.
9. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
10. Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *Proceedings of the European Conference on Computer Vision (ECCV)*.
11. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*.
12. [Makesense.ai](https://www.makesense.ai/)

