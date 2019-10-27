import cv2
import os
import numpy as np
import random


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    threshold = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > threshold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def main():
    for folder in os.listdir("../stanford-car-dataset-by-classes-folder/car_data/car_data/train")[8:]:
        folder = os.path.join("../stanford-car-dataset-by-classes-folder/car_data/car_data/train", folder)
        for file in os.listdir(folder):
            original_image = cv2.imread(os.path.join(folder, file))
            blur = cv2.blur(original_image, (3, 3))
            noise_img = sp_noise(original_image, 0.02)
            flip_horizontal = cv2.flip(original_image, 1)
            cv2.imwrite(os.path.splitext(os.path.join(folder, file))[0] + "_blur_3.jpg", blur)
            cv2.imwrite(os.path.splitext(os.path.join(folder, file))[0] + "_S&P_noise.jpg", noise_img)
            cv2.imwrite(os.path.splitext(os.path.join(folder, file))[0] + "_h_flip.jpg", flip_horizontal)


if __name__ == '__main__':
    main()
