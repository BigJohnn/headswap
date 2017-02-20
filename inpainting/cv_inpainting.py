#!/usr/bin/env python
# encoding: utf-8

# -------------------------------------------------------
# version: v0.1
# author: lirui
# license: Apache Licence
# contact: lirui_buaa@163.com
# project: dcgan-completion
# function:
# file: cv_inpainting
# time: 17-1-12 上午9:47
# ---------------------------------------------------------

import cv2
import os
import os.path as osp

import libs
import numpy as np
import time


class InpaintPair:
    def __init__(self, distorted_file, mask_file):
        if isinstance(distorted_file, str):
            self.img_distorted = cv2.imread(distorted_file)
            self.img_mask = cv2.imread(mask_file, 0)
            self._filename = osp.split(distorted_file)[-1]
        elif isinstance(distorted_file, np.ndarray):
            self.img_distorted, self.img_mask = distorted_file, mask_file
            self._filename = ""
        else:
            raise ValueError("both distorted_file and mask_file must be string or numpy.ndarray")

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not isinstance(filename, str):
            raise ValueError("filename must be an string")
        if osp.splitext(filename) not in ['.jpg', '.png']:
            raise ValueError("filename indicates a image file, and must end with right format,i.e.,'.jpg','.png'")


class CvInpainter:
    """
    基于opencv的图像修复
    """

    def __init__(self, distorted_file, mask_file):
        self._inpaint_pair = globals()["InpaintPair"](distorted_file, mask_file)

    def inpaint(self):
        return cv2.inpaint(self._inpaint_pair.img_distorted, self._inpaint_pair.img_mask, 10, cv2.INPAINT_NS)


class InpaintInterface:
    '''
    图像修复器接口程序
    '''

    @classmethod
    def img_inpaint(cls, distorted_file, mask_file):
        start_time = time.time()
        inpainted = CvInpainter(distorted_file, mask_file).inpaint()
        print (time.time() - start_time)
        return inpainted

    @classmethod
    def img_sequence_inpaint(cls, imgs_dir):
        '''
        图像序列进行一一修复
        :param imgs_dir:图像根目录，ori，mask两个目录，存放待修复图像及修复区域图像，修复完成后存入inpainted目录
        :return:
        '''
        distorted_dir, mask_dir, out_dir = map(libs.PathHelper.join_path(imgs_dir), ("ori", "mask", "inpainted"))
        libs.PathHelper.ensure_dir_exist(out_dir)
        distorted_imgs, mask_imgs = map(os.listdir, (distorted_dir, mask_dir))
        for imgfile in set(distorted_imgs) & set(mask_imgs):
            print (imgfile)
            out_filename = libs.PathHelper.join_path(out_dir)(imgfile)
            inpainted = cls.img_inpaint(*map(lambda x: osp.join(x, imgfile), (distorted_dir, mask_dir)))
            cv2.imwrite(out_filename, inpainted)

import inpainting.draw_mask

def img_sequence_roi_inpaint(imgs_dir):
    # 由ROI区域 生成 待修复的轮廓mask
    mask_dir, roi_dir = map(libs.PathHelper.join_path(imgs_dir), ("mask", "ROI"))
    libs.PathHelper.ensure_dir_exist(mask_dir)
    roi_imgs = os.listdir(roi_dir)
    for filename in roi_imgs:
        roi_file, mask_file = map(lambda d: osp.join(d, filename), (roi_dir, mask_dir))
        mask, con = inpainting.draw_mask.DrawMask.contour_inner_mask(roi_file)
        cv2.imwrite(mask_file, mask)
    InpaintInterface.img_sequence_inpaint(imgs_dir)


def render(filename, roi_file=""):
    img = cv2.imread(filename)
    print ("图像尺寸", img.shape)
    mask_file = filename.replace(".jpg", "_mask.jpg")
    contours = None
    start_time = time.time()
    if not roi_file:
        mask = inpainting.draw_mask.DrawMask.face_mask(img)
    else:
        mask, contours = inpainting.draw_mask.DrawMask.contour_inner_mask(roi_file)
    print ("获取ROI掩膜", time.time() - start_time)
    img_rendered = InpaintInterface.img_inpaint(img, mask)
    print ("背景填充ROI", time.time() - start_time)
    # cv2.drawContours(img_rendered, contours, -1, (255, 255, 255), 2)
    cv2.imwrite(filename.replace(".jpg", "_rendered.jpg"), img_rendered)

import shutil
def rename(imgs_dir):
    imgs_list=os.listdir(imgs_dir)
    for filename in imgs_list:
        name,format=osp.splitext(filename)
        print (name)
        newname=osp.join(imgs_dir,str(int(name))+format)
        shutil.move(osp.join(imgs_dir,filename),newname)

if __name__ == "__main__":
    # render('../Data/distorted/1.jpg', roi_file="../Data/ROI/1.jpg")
    ROOT = './dataset'
    MASK = ROOT + '/MASK'
    INPAINTED = ROOT + '/INPAINTED'
    if os.path.exists(MASK):
        shutil.rmtree(MASK)
        os.mkdir(MASK)
    else:
        os.mkdir(MASK)

    if os.path.exists(INPAINTED):
        shutil.rmtree(INPAINTED)
        os.mkdir(INPAINTED)
    else:
        os.mkdir(INPAINTED)

    img_sequence_roi_inpaint("./dataset")
    # InpaintInterface.img_sequence_inpaint("../Data/imgma")

    # CvInpainter('../Data/distorted/1586.png', '../Data/mask/1586.png').inpaint()
