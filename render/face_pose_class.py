import sys
import numpy as np
import cv2
import warnings
import os

caffe_root = 'C:\Projects\caffe\python\caffe'
sys.path.insert(0, caffe_root + 'python')
import caffe
import time
import dlib
import matplotlib.pyplot as plt

class FacePose(object):
    def __init__(self,model_dir):
        self.system_height = 650
        self.system_width = 1280
        self.channels = 1
        self.test_num = 1
        self.pointNum = 68

        self.S0_width = 60
        self.S0_height = 60
        self.vgg_height = 224
        self.vgg_width = 224
        self.M_left = -0.15
        self.M_right = +1.15
        self.M_top = -0.10
        self.M_bottom = +1.25
        self.pose_name = ['Pitch', 'Yaw', 'Roll']  # respect to  ['head down','out of plane left','in plane right']
        self.count = 0;
        self.model_dir = model_dir
        self.__load_model()
    def __load_model(self):
        vgg_point_MODEL_FILE = './models/deploy.prototxt'
        vgg_point_PRETRAINED = './models/68point_dlib_with_pose.caffemodel'
        mean_filename = './models/VGG_mean.binaryproto'
        self.vgg_point_net = caffe.Net(vgg_point_MODEL_FILE, vgg_point_PRETRAINED, caffe.TEST)
        # caffe.set_mode_gpu()
        # caffe.set_device(0)
        caffe.set_mode_cpu()
        proto_data = open(mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        self.mean = caffe.io.blobproto_to_array(a)[0]

    def __detectFace(self,img):
        detector = dlib.get_frontal_face_detector()
        dets = detector(img, 1)
        bboxs = np.zeros((len(dets), 4))
        for i, d in enumerate(dets):
            bboxs[i, 0] = d.left();
            bboxs[i, 1] = d.right();
            bboxs[i, 2] = d.top();
            bboxs[i, 3] = d.bottom();
        return bboxs;

    def __getCutSize(self,bbox, left, right, top, bottom):  # left, right, top, and bottom

        box_width = bbox[1] - bbox[0]
        box_height = bbox[3] - bbox[2]
        cut_size = np.zeros((4))
        cut_size[0] = bbox[0] + left * box_width
        cut_size[1] = bbox[1] + (right - 1) * box_width
        cut_size[2] = bbox[2] + top * box_height
        cut_size[3] = bbox[3] + (bottom - 1) * box_height
        return cut_size

    def __retifyBBoxSize(self,img_height, img_width, bbox):
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[2] < 0:
            bbox[2] = 0
        if bbox[3] < 0:
            bbox[3] = 0
        if bbox[0] > img_width:
            bbox[0] = img_width
        if bbox[1] > img_width:
            bbox[1] = img_width
        if bbox[2] > img_height:
            bbox[2] = img_height
        if bbox[3] > img_height:
            bbox[3] = img_height
        return bbox

    def __retifyBBox(self,img, bbox):
        img_height = np.shape(img)[0] - 1
        img_width = np.shape(img)[1] - 1
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[2] < 0:
            bbox[2] = 0
        if bbox[3] < 0:
            bbox[3] = 0
        if bbox[0] > img_width:
            bbox[0] = img_width
        if bbox[1] > img_width:
            bbox[1] = img_width
        if bbox[2] > img_height:
            bbox[2] = img_height
        if bbox[3] > img_height:
            bbox[3] = img_height
        return bbox

    def __getRGBTestPart(self,bbox, left, right, top, bottom, img, height, width):
        largeBBox = self.__getCutSize(bbox, left, right, top, bottom)
        retiBBox = self.__retifyBBox(img, largeBBox)
        # cv2.rectangle(img, (int(retiBBox[0]), int(retiBBox[2])), (int(retiBBox[1]), int(retiBBox[3])), (0,0,255), 2)
        # cv2.imshow('f',img)
        # cv2.waitKey(0)
        face = img[int(retiBBox[2]):int(retiBBox[3]), int(retiBBox[0]):int(retiBBox[1]), :]

        # cv2.imshow('f', face)
        # cv2.waitkey(0)
        face = cv2.resize(face, (height, width), interpolation=cv2.INTER_AREA)
        face = face.astype('float32')
        return face

    def __batchRecoverPart(self,predictPoint, totalBBox, totalSize, left, right,  top, bottom, height, width):
        recoverPoint = np.zeros(predictPoint.shape)
        for i in range(0, predictPoint.shape[0]):
            recoverPoint[i] = self.recoverPart(predictPoint[i], totalBBox[i], left, right, top, bottom, totalSize[i, 0],
                                          totalSize[i, 1], height, width)
        return recoverPoint

    def show_image(img, facepoint, bboxs, headpose):
        plt.figure(figsize=(20, 10))
        for faceNum in range(0, facepoint.shape[0]):
            cv2.rectangle(img, (int(bboxs[faceNum, 0]), int(bboxs[faceNum, 2])),
                          (int(bboxs[faceNum, 1]), int(bboxs[faceNum, 3])), (0, 0, 255), 2)
            for p in range(0, 3):
                plt.text(int(bboxs[faceNum, 0]), int(bboxs[faceNum, 2]) - p * 30,
                         '{:s} {:.2f}'.format(pose_name[p], headpose[faceNum, p]),
                         bbox=dict(facecolor='blue', alpha=0.5),
                         fontsize=12, color='white')
            for i in range(0, facepoint.shape[1] / 2):
                cv2.circle(img, (int(round(facepoint[faceNum, i * 2])), int(round(facepoint[faceNum, i * 2 + 1]))), 1,
                           (0, 255, 0), 2)
        height = img.shape[0]
        width = img.shape[1]
        # if height > system_height or width > system_width:
        #     height_radius = system_height * 1.0 / height
        #     width_radius = system_width * 1.0 / width
        #     radius = min(height_radius, width_radius)
        #     img = cv2.resize(img, (0, 0), fx=radius, fy=radius)

        img = img[:, :, [2, 1, 0]]
        plt.imshow(img)
        plt.show()

    def predictImage(self,colorImage):
        bboxs = self.__detectFace(colorImage)
        faceNum = bboxs.shape[0]
        faces = np.zeros((1, 3, self.vgg_height, self.vgg_width))
        predictpoints = np.zeros((faceNum, self.pointNum * 2))
        predictpose = np.zeros((faceNum, 3))
        imgsize = np.zeros((2))
        imgsize[0] = colorImage.shape[0] - 1
        imgsize[1] = colorImage.shape[1] - 1
        TotalSize = np.zeros((faceNum, 2))
        for i in range(0, faceNum):
            TotalSize[i] = imgsize
        for i in range(0, faceNum):
            bbox = bboxs[i]
            colorface = self.__getRGBTestPart(bbox, self.M_left, self.M_right, self.M_top, self.M_bottom, colorImage, self.vgg_height, self.vgg_width)
            normalface = np.zeros(self.mean.shape)
            normalface[0] = colorface[:, :, 0]
            normalface[1] = colorface[:, :, 1]
            normalface[2] = colorface[:, :, 2]
            normalface = normalface - self.mean
            faces[0] = normalface
            blobName = '68point'
            data4DL = np.zeros([faces.shape[0],1,1,1])
            self.vgg_point_net.set_input_arrays(faces.astype(np.float32),data4DL.astype(np.float32))
            self.vgg_point_net.forward()
            predictpoints[i] = self.vgg_point_net.blobs[blobName].data[0]
            blobName = 'poselayer'
            pose_prediction = self.vgg_point_net.blobs[blobName].data
            predictpose[i] = pose_prediction * 50
       #predictpoints = predictpoints * self.vgg_height/2 + self.vgg_width/2
        # level1Point = self.batchRecoverPart(predictpoints,bboxs,TotalSize,self.M_left,self.M_right,self.M_top,self.M_bottom,self.vgg_height,self.vgg_width)
        # self.show_image(colorImage, level1Point, bboxs, predictpose)
        print (predictpose)

        try:
            return -predictpose[0][0],-predictpose[0][1],-predictpose[0][2]
        except:
            return None,None,None

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    FM=FacePose('./models')
    # flie_list = os.listdir(r"/home/haojun/head-pose-estimation-and-face-landmark-master/img")
    # for img_name in flie_list:
    #     img = cv2.imread(r"/home/haojun/head-pose-estimation-and-face-landmark-master/img" + '/' + img_name)
    #     Pitch, Yaw, Roll = FM.predictImage(img)
    #     print (Pitch, Yaw, Roll)
    # print (Pitch,Yaw,Roll)
    img = cv2.imread('1.jpg')
    Pitch, Yaw, Roll = FM.predictImage(img)


