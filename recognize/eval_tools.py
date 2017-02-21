from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from recognize import detect_face
import time
import cv2


class Compare(object):
    def __init__(self, model_dir, image_size=160, margin=44, gpu_memory_fraction=0.4):
        self.image_size = image_size
        self.margin = margin
        self.gpu_memory_fraction = gpu_memory_fraction
        self.model_dir = model_dir

        # Load the model
        # print('Model directory: %s' % self.model_dir)
        # meta_file, ckpt_file = self.__get_model_filenames(model_dir)
        # print('Metagraph file: %s' % meta_file)
        # print('Checkpoint file: %s' % ckpt_file)
        print('Creating networks and loading parameters...')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)

            self.__sess_feature = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                                   log_device_placement=False,
                                                                   allow_soft_placement=True
                                                                   ))
            self.__graph_def = tf.GraphDef()
            self.__load_model("./models/model.pb")
            # Get input and output tensors
            self.__images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.__embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.__phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    log_device_placement=False,
                                                    allow_soft_placement=True
                                                    ))
            with sess.as_default():
                self.__pnet, self.__rnet, self.__onet = detect_face.create_mtcnn(sess, self.model_dir)
        print("init success..")

    def __load_model(self, pb_model_path):
        with open(pb_model_path, "rb") as f:
            self.__graph_def.ParseFromString(f.read())
            for node in self.__graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] += '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr:
                        del node.attr['use_locking']
            _ = tf.import_graph_def(self.__graph_def, name="")

    def eval_dist(self, image1, image2):
        init_imagelist = [image1, image2]
        images = self.load_and_align_data(init_imagelist)
        result = self.__extract_calculate(images)
        return result

    def load_and_align_data(self, init_images):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        nrof_samples = len(init_images)
        img_list = [None] * nrof_samples
        for i in range(nrof_samples):
            # img = misc.imread(os.path.expanduser(image_paths[i]))

            img_size = np.asarray(init_images[i].shape)[0:2]
            bounding_boxes, _ = detect_face.detect_face(init_images[i], minsize,
                                                        self.__pnet, self.__rnet, self.__onet,
                                                        threshold, factor)

            if np.size(bounding_boxes) != 0:
                det = np.squeeze(bounding_boxes[0, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                cropped = init_images[i][bb[1]:bb[3], bb[0]:bb[2], :]
            else:
                cropped = init_images[i]

            cropped = cv2.resize(cropped, (self.image_size, self.image_size))

            pre_whitened = self.__pre_whiten(cropped)
            img_list[i] = pre_whitened
        images = np.stack(img_list)
        t02 = time.clock()
        # print("align time for 2 images: %fs" % (t02 - t01))
        return images

    def __extract_calculate(self, images):
        # Run forward pass to calculate embeddings
        feed_dict = {self.__images_placeholder: images, self.__phase_train_placeholder: False}

        emb = self.__sess_feature.run(self.__embeddings, feed_dict=feed_dict)
        # print(np.shape(emb))

        # print("extract feature time: %fs" % (t04 - t03))
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))

        return dist

    @staticmethod
    def __pre_whiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def eval_in_class(self, imgs_dir):
        imglist = os.listdir(imgs_dir)
        res_list = []
        for i in range(len(imglist) - 1):
            for j in range(i + 1, len(imglist) - 1):
                img_i = cv2.imread(imgs_dir + "%04d" % i + ".jpg")
                img_j = cv2.imread(imgs_dir + "%04d" % j + ".jpg")

                x = self.eval_dist(img_i, img_j)
                print("%02d and %02d result: %f" % (i, j, x))
                if x < 1.0:
                    res_list.append(x)
        res_nol = np.sum(res_list)
        print(res_nol)
        return res_nol

    def eval_btw_class(self, class1_dir, class2_dir):
        img_list1 = os.listdir(class1_dir)
        img_list2 = os.listdir(class2_dir)
        res_list = []
        for i in range(len(img_list1) - 1):
            for j in range(len(img_list2) - 1):
                img_i = cv2.imread(class1_dir + "%04d" % i + ".jpg")
                img_j = cv2.imread(class2_dir + "%04d" % j + ".jpg")
                x = self.eval_dist(img_i, img_j)
                print("%02d and %02d result: %f" % (i, j, x))
                if x < 1.0:
                    res_list.append(x)
        res_nol = np.sum(res_list)
        print(res_nol)
        return res_nol

    def extract_single_feature(self, image):
        feed_dict = {self.__images_placeholder: [image]}
        feature = self.__sess_feature.run(self.__embeddings, feed_dict=feed_dict)
        return feature

    def detect_facerect(self, image):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        bounding_boxes, _ = detect_face.detect_face(image, minsize,
                                                    self.__pnet, self.__rnet, self.__onet,
                                                    threshold, factor)

        nums = len(bounding_boxes)
        img_size = np.asarray(image.shape)[0:2]
        rects = []
        faces = []
        for i in range(nums):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            rects.append(bb)

            cropped = cv2.resize(cropped, (self.image_size, self.image_size))
            pre_whitened = self.__pre_whiten(cropped)
            faces.append(pre_whitened)
        return rects, faces


def test_face_detect():
    cp = Compare('D:/work/compare/models')
    img = cv2.imread("./1523.jpg")
    for _ in range(50):
        rect, face = cp.detect_facerect(img)


if __name__ == '__main__':
    cp = Compare('D:/work/compare/models')
    data_path = "./data/actor_face/"
    img1 = cv2.imread(data_path + "fanbingbing" + "/" + "%04d" % 10 + ".jpg")
    img1 = cv2.resize(img1, (160, 160))
    img2 = cv2.imread(data_path + "fanbingbing" + "/" + "%04d" % 11 + ".jpg")
    img2 = cv2.resize(img2, (160, 160))

    for i in range(50):
        t1 = time.clock()
        score = cp.eval_dist(img1, img2)
        print("time: %s, score: %f" % ((time.clock() - t1), score))

        # test_face_detect()
