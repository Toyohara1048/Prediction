import numpy as np
import cv2
import os
import math

LOCAL = False

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


def load_PC(num_of_data):
    """
    Loads primary components from entire data sets
    :param num_of_data: The max number(name) of 5-frame sequence data to load
    :return: np.array (shape: (the true number of data, NUM_OF_FRAMES, 3))
    """


# 以下実験
# data = load_5frames(3)
# img = data[0]
# # nLabels, labeledImage, stats, centeroids = cv2.connectedComponentsWithStats(img)
# # print(nLabels)
# # print(centeroids)
# # print(stats)
# # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# # cv2.circle(img, (int(centeroids[2][0]), int(centeroids[2][1])), 5, (255, 0, 0), -1)
# def drawAxis(img, start_pt, vec, colour, length):
#     # アンチエイリアス
#     CV_AA = 16
#
#     # 終了点
#     end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))
#
#     # 中心を描画
#     cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)
#
#     # 軸線を描画
#     cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);
#
#     # 先端の矢印を描画
#     angle = math.atan2(vec[1], vec[0])
#     print(angle)
#     print(vec[0])
#     print(vec[1])
#     print(vec[0]*vec[0] + vec[1]*vec[1])
#
#     qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4));
#     qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4));
#     cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA);
#
#     qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4));
#     qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4));
#     cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA);
#
# img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
# # 各輪郭に対する処理
# for i in range(0, len(contours)):
#
#     # 輪郭の領域を計算
#     area = cv2.contourArea(contours[i])
#
#     # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
#     if area < 1e2 or 1e5 < area:
#         continue
#
#     # 輪郭を描画する
#     cv2.drawContours(img, contours, i, (0, 0, 255), 2, 8, hierarchy, 0)
#
#     # 輪郭データを浮動小数点型の配列に格納
#     X = np.array(contours[i], dtype=np.float).reshape((contours[i].shape[0], contours[i].shape[2]))
#
#     # PCA（１次元）
#     mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float), maxComponents=1)
#
#     # 主成分方向のベクトルを描画
#     pt = (mean[0][0], mean[0][1])
#     vec = (eigenvectors[0][0], eigenvectors[0][1])
#     drawAxis(img, pt, vec, (255, 255, 0), 150)
#
# cv2.imshow("test", img)
# cv2.waitKey(0)

