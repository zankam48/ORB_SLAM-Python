import os
import cv2
from utils import read_image_file_list, load_images


def extract_orb_features(images):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints_list = []
    descriptors_list = []
    for timestamp, image in images:

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        keypoints_list.append((timestamp, keypoints))
        descriptors_list.append((timestamp, descriptors))
    return keypoints_list, descriptors_list

def visualize_keypoints(images, keypoints_list):
    for (timestamp, image), (_, keypoints) in zip(images, keypoints_list):
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imshow(f'ORB Keypoints - Timestamp: {timestamp}', image_with_keypoints)
        key = cv2.waitKey(0)
        if key == 27: 
            break
    cv2.destroyAllWindows()

def main():
    dataset_dir = 'D:\Coding Project\markerless ar with orb slam\data\\fr1_desk'  # Replace with the actual path

    rgb_file_list_path = os.path.join(dataset_dir, 'rgb.txt')

    rgb_image_file_list = read_image_file_list(rgb_file_list_path)

    rgb_images = load_images(dataset_dir, rgb_image_file_list)

    keypoints_list, descriptors_list = extract_orb_features(rgb_images)

    visualize_keypoints(rgb_images, keypoints_list)

if __name__ == '__main__':
    main()
