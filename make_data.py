import cv2
import numpy as np

ORG_FILE_NAME = "videos/data.avi"


FRAME_RATE = 40
DEPTH = 5
LENGTH_OF_SIDE = 488

start_num = 141

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
    cropped = c_frame[0:LENGTH_OF_SIDE, 80:80+LENGTH_OF_SIDE]
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
for i in range(start_num, start_num+frame_count-10):
    rec = cv2.VideoWriter("videos/" + str(i) + ".avi", fourcc, FRAME_RATE, (LENGTH_OF_SIDE, LENGTH_OF_SIDE))
    for j in range(DEPTH):
        rec.write(video[i-start_num+j, 0:LENGTH_OF_SIDE, 0:LENGTH_OF_SIDE, 0:3])
    rec.release()

# 終了処理
cv2.destroyAllWindows()
org.release()
