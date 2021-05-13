# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np

import paint_black

# フレーム差分の計算
def frame_sub(img1, img2, img3, th):
    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

    # 二値化処理
    # diff[diff < th] = 0
    # diff[diff >= th] = 255
    
    # メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 3)
    mask = cv2.medianBlur(mask, 3)

    return  mask


# 動画ファイルのキャプチャ
cap = cv2.VideoCapture("20191201F-netvsYSCC.mp4")
cap2 = cv2.VideoCapture("20191201F-netvsYSCC.mp4")
cap3 = cv2.VideoCapture("20191201F-netvsYSCC.mp4")
    
# フレームを3枚取得してグレースケール変換
# frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
frame1 = cap.read()[1]

# 50フレーム差の差分
for i in range(50):
    cap2.read()
# frame2 = cv2.cvtColor(cap2.read()[1], cv2.COLOR_RGB2GRAY)
frame2 = cap2.read()[1]

for i in range(100):
    cap3.read()
# frame3 = cv2.cvtColor(cap3.read()[1], cv2.COLOR_RGB2GRAY)
frame3 = cap3.read()[1]

while(cap3.isOpened()):
    frame1 = paint_black.paint_black(frame1)
    frame2 = paint_black.paint_black(frame2)
    frame3 = paint_black.paint_black(frame3)
    # フレーム間差分を計算
    mask = frame_sub(frame1, frame2, frame3, th=20)

    # 結果を表示
    cv2.imshow("Frame2", frame2)
    cv2.imshow("Mask", mask)

    # 3枚のフレームを更新
    # frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame1 = cap.read()[1]    
    # 50フレーム差の差分
    # frame2 = cv2.cvtColor(cap2.read()[1], cv2.COLOR_RGB2GRAY)
    # frame3 = cv2.cvtColor(cap3.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cap2.read()[1]
    frame3 = cap3.read()[1]


    # 待機(0.03sec)
    time.sleep(0.03)

    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()