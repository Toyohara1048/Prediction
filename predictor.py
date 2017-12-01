import numpy as np
import math
import os
import tensorflow as tf
from PIL import Image
import cv2

from keras.models import load_model

from load_video import load_PC, load_5frames


LOCAL = True

# Weight saving
WEIGHT_PATH = '/media/hdd/Toyohara/PredictNextPose/weight/'
LOCAL_WEIGHT_PATH = 'weight/'


def combine_images(generated_images):
    total = generated_images.shape[0]
    cols = total
    rows = 1
    width = generated_images.shape[1]
    height = generated_images.shape[2]
    combined_image = np.zeros((int(height*rows), int(width*cols), 3),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1), 0:3] = image[:, :, 0:3]
    return combined_image


def prediction(index):
    if LOCAL:
        path = LOCAL_WEIGHT_PATH
    else:
        path = WEIGHT_PATH

    model = load_model(path+'predictor.h5')
    model.load_weights(path+'predictor_weight.h5')

    # Data load
    centers_X, centers_Y, tans_X, tans_Y = load_PC()

    # Prediction
    center, tan = model.predict({'center_in': centers_X, 'tan_in': tans_X})

    images = load_5frames(index)
    result = np.array([], dtype=np.uint8)
    for i in range(4):
        temp = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        cv2.circle(temp, (int(centers_X[index][2*i] * temp.shape[0]), int(centers_X[index][2*i+1] * temp.shape[1])), 10, (255, 0, 0), -1)
        rad = math.atan(tans_X[index][i])
        end_pt = (int(centers_X[index][0] * temp.shape[0] * 50 * math.cos(rad)), int(centers_X[index][1] * temp.shape[1] * 50 * math.sin(rad)))
        cv2.line(temp, (int(centers_X[index][2*i] * temp.shape[0]), int(centers_X[index][2*i+1] * temp.shape[1])), end_pt, (255, 255, 0), 1, 16);
        result = np.append(result, temp)

    temp = cv2.cvtColor(images[4], cv2.COLOR_GRAY2BGR)
    cv2.circle(temp, (int(center[index][0] * temp.shape[0]), int(center[index][1] * temp.shape[1])), 10, (0, 0, 255), -1)
    rad = math.atan(tan[index][0])
    end_pt = (int(center[index][0] * temp.shape[0] * 50 * math.cos(rad)),
              int(center[index][1] * temp.shape[1] * 50 * math.sin(rad)))
    cv2.line(temp, (int(center[index][0] * temp.shape[0]), int(center[index][1] * temp.shape[1])),
             end_pt, (0, 255, 255), 1, 16);
    result = np.append(result, temp)
    result = result.reshape(5, 488, 488, 3)

    result_image = combine_images(result)
    quarter = cv2.resize(result_image, None, fx=1 / 2, fy=1 / 2)
    cv2.imshow("Result", quarter)
    cv2.waitKey(0)



    print(center[index], tan[index])

if __name__ == "__main__":
    prediction(0)