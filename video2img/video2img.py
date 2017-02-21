import cv2
import os
import shutil

SAVE_PATH = "./dataset/ori/"
def run(video_path, save_path=SAVE_PATH):

    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        os.mkdir(SAVE_PATH)
    else:
        os.mkdir(SAVE_PATH)

    cap = cv2.VideoCapture()
    cap.open(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 92496)
    num = 0
    while True:
        mark, img = cap.read()
        img = cv2.flip(img,-1)
        # cv2.imshow('img',img)
        # cv2.waitKey()
        if not mark:
            break
        cv2.imwrite(save_path+"%04d.jpg" % num, img)
        # cv2.waitKey(1)
        num += 1
    print("all image saved")
