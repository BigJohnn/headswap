import cv2
import numpy as np
from recognize import eval_tools as et
import os
import inpainting

class Mask:
    def __init__(self, model_path="./models", target_person="./recognize/face_lib/person1.png"):
        print("init recognize model, pls wait...")
        self.cp = et.Compare(model_path)
        self.target_img = cv2.imread(target_person)
        self.skip_count = 0
        self.process_num = 0

    def save_pic(self, seg, ori):
        mask = np.zeros_like(seg)
        rects, faces = self.cp.detect_facerect(ori)


        # PS: check which face for swap
        if len(faces) >= 1:
            index = 0
            for k, face in enumerate(faces):
                score = self.cp.eval_dist(face, self.target_img)
                if score < 1.0:
                    index = k

            roi = rects[index]
            # cv2.rectangle(ori, (roi[0], roi[1]), (roi[2], roi[3]), [0, 255, 255])
            # cv2.imshow("detect", ori)
            # cv2.waitKey(0)

            # mask[(seg[:, :] == [0, 0, 128]).all(axis=2)] = 255
            # PS: color the region of interest
            mask[roi[1]:roi[3], roi[0]:roi[2]][(seg[roi[1]:roi[3], roi[0]:roi[2]] == [0, 0, 128]).all(axis=2)] = 255
            ori[roi[1]:roi[3], roi[0]:roi[2]][(seg[roi[1]:roi[3], roi[0]:roi[2]] == [0, 0, 128]).all(axis=2)] = 255
            # print(np.shape((seg[:, :] == [0, 0, 128]).all(axis=2)))
            # [(seg[:, :] == [0, 0, 128]).all(axis=2)]
            # cv2.imshow("test", ori)

        else:
            print("no face detected, skip")
            mask[(seg[:, :] == [0, 0, 128]).all(axis=2)]=255
            ori[(seg[:, :] == [0, 0, 128]).all(axis=2)]=255
            self.skip_count += 1

        cv2.imwrite("./dataset/background/%04d.jpg" % self.process_num, ori)
        cv2.imwrite("./dataset/ROI/%04d.jpg" % self.process_num, mask)
        cv2.waitKey(1)
        self.process_num += 1


def run():
    ROOT = './dataset'
    ROI = ROOT + '/ROI'
    BACKGROUND = ROOT + '/background'

    import shutil
    if os.path.exists(ROI):
        shutil.rmtree(ROI)
        os.mkdir(ROI)
    else:
        os.mkdir(ROI)
    if os.path.exists(BACKGROUND):
        shutil.rmtree(BACKGROUND)
        os.mkdir(BACKGROUND)
    else:
        os.mkdir(BACKGROUND)

    mk = Mask()
    for i in range(len(os.listdir("./dataset/ori"))):
        seg = cv2.imread("./dataset/seg/predict_result_mask/%04d.png" % i)
        ori = cv2.imread("./dataset/ori/%04d.jpg" % i)
        mk.save_pic(seg, ori)
        if i % 100 == 0:
            print("%04d success..." % i)

if __name__ == "__main__":
    run()
