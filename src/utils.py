import os
import cv2

def read_image_file_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    image_files = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue  
        parts = line.strip().split()
        timestamp = float(parts[0])
        filename = parts[1]
        image_files.append((timestamp, filename))
    return image_files

def load_images(image_dir, image_file_list):
    images = []
    for timestamp, filename in image_file_list:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append((timestamp, image))
        else:
            print(f"Warning: Could not load image {image_path}")
    return images

