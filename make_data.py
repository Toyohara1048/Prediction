import cv2
import numpy as np
import os
import math

ORG_FILE_NAME = "videos/data.avi"


FRAME_RATE = 40
NUM_OF_FRAMES = 5
LENGTH_OF_SIDE = 488

start_num = 141

def make_5frames():
    # 元ビデオファイル読み込み
    org = cv2.VideoCapture(ORG_FILE_NAME)

    # 保存ビデオファイルの準備
    end_flag, c_frame = org.read()
    height, width, channels = c_frame.shape
    print("The shape of video is:")
    print(c_frame.shape)

    # 動画の読み込み
    video = np.array([], dtype=np.uint8)
    frame_count = 0
    while end_flag is True:
        # cropping
        cropped = c_frame[0:LENGTH_OF_SIDE, 80:80 + LENGTH_OF_SIDE]
        video = np.append(video, np.array(cropped[0:LENGTH_OF_SIDE, 0:LENGTH_OF_SIDE]))

        # 次のフレーム読み込み
        end_flag, c_frame = org.read()
        frame_count += 1

        if frame_count % 100 is 0:
            print("%d frames loaded" % frame_count)

    video = np.reshape(video, (frame_count, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 3))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # 指定フレームごとに切り分け
    for i in range(start_num, start_num + frame_count - 10):
        rec = cv2.VideoWriter("videos/" + str(i) + ".avi", fourcc, FRAME_RATE, (LENGTH_OF_SIDE, LENGTH_OF_SIDE))
        for j in range(NUM_OF_FRAMES):
            rec.write(video[i - start_num + j, 0:LENGTH_OF_SIDE, 0:LENGTH_OF_SIDE, 0:3])
        rec.release()

    # 終了処理
    cv2.destroyAllWindows()
    org.release()


def drawAxis(img, start_pt, vec, colour, length):
    # アンチエイリアス
    CV_AA = 16

    # 終了点
    end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

    # 中心を描画
    cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)

    # 軸線を描画
    cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);

    # 先端の矢印を描画
    angle = math.atan2(vec[1], vec[0])
    # print(angle)
    # print(vec)
    # print(vec[1])
    # print(vec[0]*vec[0] + vec[1]*vec[1])

    qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4));
    qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4));
    cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA);

    qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4));
    qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4));
    cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA);


def PCA(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    temp, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # sort by area
    index_and_area = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 1e2:
            index_and_area.append((i, area))
    index_and_area.sort(key=lambda taple: taple[1], reverse=True)
    cv2.drawContours(rgb, contours, index_and_area[0][0], (0, 0, 255), 2, 8, hierarchy, 0)

    # 輪郭データを浮動小数点型の配列に格納
    X = np.array(contours[index_and_area[0][0]], dtype=np.float).reshape((contours[index_and_area[0][0]].shape[0], contours[index_and_area[0][0]].shape[2]))

    # PCA（１次元）
    mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float), maxComponents=1)

    # 主成分方向のベクトルを描画
    pt = (mean[0][0], mean[0][1])
    vec = (eigenvectors[0][0], eigenvectors[0][1])
    tan = eigenvectors[0][1] / eigenvectors[0][0]
    # drawAxis(rgb, pt, vec, (255, 255, 0), 150)
    # cv2.imshow("test", rgb)
    # cv2.waitKey(0)

    return pt, tan


def PCA_on_5frame(name):
    frames = np.array([], dtype=np.uint8)
    centers = np.array([], dtype=np.float32)
    tans = np.array([], dtype=np.float32)

    file_name = "videos/" + str(name) + ".avi"
    if not os.path.exists(file_name):
        return None, None

    org = cv2.VideoCapture(file_name)
    end_flag, c_frame = org.read()

    while end_flag == True:
        gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        frames = np.append(frames, np.array(gray[0:488, 0:488]))

        # 次のフレーム読み込み
        end_flag, c_frame = org.read()

    frames = frames.reshape(NUM_OF_FRAMES, LENGTH_OF_SIDE, LENGTH_OF_SIDE, 1)

    for i in range(NUM_OF_FRAMES):
        center, tan = PCA(frames[i][0:LENGTH_OF_SIDE, 0:LENGTH_OF_SIDE])
        centers = np.append(centers, np.array((center[0]/LENGTH_OF_SIDE, center[1]/LENGTH_OF_SIDE)))
        tans = np.append(tans, np.array((tan)))

    return centers, tans


def make_PC_on_all(num):
    centers_X = np.array([], dtype=np.float32)
    centers_Y = np.array([], dtype=np.float32)
    tans_X = np.array([], dtype=np.float32)
    tans_Y = np.array([], dtype=np.float32)

    count = 0
    for i in range(num):
        center, tan = PCA_on_5frame(i)
        if center is not None:
            centers_X = np.append(centers_X, center[0:8])
            centers_Y = np.append(centers_Y, center[8:10])
            tans_X = np.append(tans_X, tan[0:4])
            tans_Y = np.append(tans_Y, tan[4:5])
            count += 1

    centers_X = centers_X.reshape(count, 8)
    centers_Y = centers_Y.reshape(count, 2)
    tans_X = tans_X.reshape(count, 4)
    tans_Y = tans_Y.reshape(count, 1)


    np.savez('PCA_result.npz', center_X=centers_X, center_Y=centers_Y, tan_X=tans_X, tan_Y=tans_Y)

    print(tans_X.shape)


if __name__ == "__main__":
    #make_5frames()
    #PCA_on_5frame(2)
    make_PC_on_all(3192)
