import numpy as np
import cv2
import os

LOCAL = True

NUM_OF_FRAMES = 5
LENGTH_OF_SIDE = 122


def load_5frames(name):
    """
    Loads one sequence data (5 frames)
    :param name: the name of the 5-frame sequence data (e.g. 4.avi)
    :return: np.array of 5 frames data (shape: (NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    """

    train = np.array([], dtype=np.uint8)

    if LOCAL:
        file_name = "videos/" + str(name) + ".avi"
    else:
        file_name = "/media/hdd/Toyohara/PredictNextPose/videos/" + str(name) + ".avi"
    if not os.path.exists(file_name):
        return None

    org = cv2.VideoCapture(file_name)
    end_flag, c_frame = org.read()

    while end_flag == True:

        gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)

        quarter = cv2.resize(gray, None, fx=1/4, fy=1/4)

        #train = np.append(train, np.array(gray[0:488, 0:488]))
        train = np.append(train, np.array(quarter[0:122, 0:122]))

        # 次のフレーム読み込み
        end_flag, c_frame = org.read()

    train = train.reshape(NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)
    return train


def load_data(num_of_data):
    """
    Loads the entire dataset
    :param num_of_data: the number of 5-frame sequence data to load
    :return: np.array (shape: (num_of_data, NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    """

    print("Loading %d data" % num_of_data)

    data = np.array([], dtype=np.uint8)
    count = 0
    for i in range(num_of_data):
        loaded_5frame = load_5frames(i)
        if loaded_5frame is not None:
            data = np.append(data, load_5frames(i))
            count += 1

    data = data.reshape(count, NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)

    return data


#
# data = load_5frames(0)
# img = data[0]
# nLabels, labeledImage, stats, centeroids = cv2.connectedComponentsWithStats(img)
# print(nLabels)
# print(centeroids)
# print(stats)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.circle(img, (int(centeroids[2][0]), int(centeroids[2][1])), 5, (255, 0, 0), -1)
# cv2.imshow("test", img)
# cv2.waitKey(0)


# height, width = img.shape[:2]
# size = (height/2, width/2)
# halfImg = cv2.resize(img, None, fx=1/4, fy=1/4)
# cv2.imshow("half", halfImg)
# print(halfImg.shape)
# cv2.waitKey(0)