import numpy as np
import cv2


NUM_OF_FRAMES = 5
LENGTH_OF_SIDE = 488


def load_5frames(name):
    """
    Loads one sequence data (5 frames)
    :param name: the name of the 5-frame sequence data (e.g. 4.avi)
    :return: np.array of 5 frames data (shape: (NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1))
    """

    train = np.array([], dtype=np.uint8)

    file_name = "videos/"+str(name)+".avi"
    org = cv2.VideoCapture(file_name)
    end_flag, c_frame = org.read()

    while end_flag == True:

        gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)

        train = np.append(train, np.array(gray[0:488, 0:488]))

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
    for i in range(num_of_data):
        data = np.append(data, load_5frames(i))

    data = data.reshape(num_of_data, NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)

    return data

# test
# data = load_data(6)
# test = data[1][0][1]
# print(test.shape)
# cv2.imshow("test", test)
# cv2.waitKey(0)