#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------------------
# version: v0.1
# author: lirui
# license: Apache Licence
# contact: lirui_buaa@163.com
# project: ImagePaint
# function:
# file: draw_mask
# time: 17-1-18 下午5:31
# ---------------------------------------------------------

import cv2
import numpy as np


class DrawMask:
    pass

    @classmethod
    def contour_inner_mask(cls, roi_file):
        roi = cv2.imread(roi_file, 0)
        img_mask1 = np.zeros(roi.shape[:], dtype=np.uint8)
        ret, th_mask = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(np.copy(th_mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_mask1, contours, -1, (255, 255, 255), 10)
        # inner_mask = cv2.bitwise_and(th_mask, img_mask1)
        outer_mask = cv2.bitwise_or(th_mask, img_mask1)
        return outer_mask, contours

    @classmethod
    def face_mask(cls, img):
        classifier_path = "./inpainting/haarcascade_frontalface_alt.xml"

        faceDetect = cv2.CascadeClassifier(classifier_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print "视频尺寸：", gray.shape
        # 人脸检测
        mask = np.zeros(img.shape[:2]).astype(np.uint8)
        faces = faceDetect.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=2,
                minSize=(30, 30),
                flags=0
        )
        # print "人脸个数", faces
        scale = 0.9
        labels = np.zeros((1, len(faces)))
        for face_rect in faces:
            x, y, w, h = face_rect
            center_x, center_y = map(np.int32, [(x + w / 2), (y + h / 2)])
            radius = np.int32((w + h) / 4 * scale)
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 5)  # 6
            cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), 5)

        return mask


if __name__ == "__main__":
    pass
