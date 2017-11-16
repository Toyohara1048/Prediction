import cv2
import numpy as np

ORG_FILE_NAME = "videos/gray1-backWhite.3gp"


FRAME_RATE = 40

# 元ビデオファイル読み込み
org = cv2.VideoCapture(ORG_FILE_NAME)

# 保存ビデオファイルの準備
end_flag, c_frame = org.read()
height, width, channels = c_frame.shape
# rec = cv2.VideoWriter(GRAY_FILE_NAME, \
#                       cv2.VideoWriter_fourcc(*'MJPG'), \
#                       FRAME_RATE, \
#                       (width, height), \
#                       False)

print(width)
print(height)

name = 21

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
rec = cv2.VideoWriter("videos/"+str(name)+".avi",fourcc, FRAME_RATE, (488,488))

# 変換処理ループ
count = 0
while end_flag == True:
    # cropping
    cropped = c_frame[0:488, 80:568]

    # フレーム表示
    cv2.imshow("Original", cropped)

    if (count >= name and count < name+5):
        rec.write(cropped)



    # Qキーで終了
    key = cv2.waitKey(1)
    if key == 0x71:
        break

    # 次のフレーム読み込み
    end_flag, c_frame = org.read()
    count+=1

print(cropped.shape)

# 終了処理
cv2.destroyAllWindows()
org.release()
rec.release()
