import cv2

def run(video_path, save_path="./dataset/ori/"):
    cap = cv2.VideoCapture()
    cap.open(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 92496)
    num = 0
    while True:
        mark, img = cap.read()
        if not mark:
            break
        cv2.imwrite(save_path+"%04d.jpg" % num, img)
        # cv2.waitKey(1)
        num += 1
    print("all image saved")
