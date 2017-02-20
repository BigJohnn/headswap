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
        print('Model directory: %s' % self.model_dir)
        meta_file, ckpt_file = self.__get_model_filenames()
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        print('Creating networks and loading parameters...')
        with tf.Graph().as_default():

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)

            self.__sess_feature = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                                   log_device_placement=False,
                                                                   allow_soft_placement=True
                                                                   ))
            self.__load_model(meta_file, ckpt_file)

            # Get input and output tensors
            self.__images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.__embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    log_device_placement=False,
                                                    allow_soft_placement=True
                                                    ))
            with sess.as_default():
                self.__pnet, self.__rnet, self.__onet = detect_face.create_mtcnn(sess, self.model_dir)
        print("init success..")

    def __load_model(self, meta_file, ckpt_file):
        model_dir_exp = os.path.expanduser(self.model_dir)
        saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
        saver.restore(self.__sess_feature, os.path.join(model_dir_exp, ckpt_file))

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
                # bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                # bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                # bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                # bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                bb[0] = np.maximum(det[0] - (det[2] - det[0]) / 3, 0)
                bb[1] = np.maximum(det[1] - (det[3] - det[1]) / 3, 0)
                bb[2] = np.minimum(det[2] + (det[2] - det[0]) / 3, img_size[1])
                bb[3] = np.minimum(det[3] + (det[3] - det[1]) / 3, img_size[0])
                cropped = init_images[i][bb[1]:bb[3], bb[0]:bb[2], :]
            else:
                cropped = init_images[i]
            # aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
            cropped = cv2.resize(cropped, (self.image_size, self.image_size))

            pre_whitened = self.__pre_whiten(cropped)
            img_list[i] = pre_whitened
        images = np.stack(img_list)
        t02 = time.clock()
        # print("align time for 2 images: %fs" % (t02 - t01))
        return images

    def __extract_calculate(self, images):
        # Run forward pass to calculate embeddings
        feed_dict = {self.__images_placeholder: images}

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

    def __get_model_filenames(self):
        files = os.listdir(self.model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % self.model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % self.model_dir)
        meta_file = meta_files[0]
        ckpt_files = [s for s in files if 'ckpt' in s]
        if len(ckpt_files) == 0:
            raise ValueError('No checkpoint file found in the model directory (%s)' % self.model_dir)
        elif len(ckpt_files) == 1:
            ckpt_file = ckpt_files[0]
        else:
            ckpt_iter = [(s, int(s.split('-')[-1])) for s in ckpt_files if 'ckpt' in s]
            sorted_iter = sorted(ckpt_iter, key=lambda tup: tup[1])
            ckpt_file = sorted_iter[-1][0]
        return meta_file, ckpt_file

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
        minsize = 50  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        # t1 = time.clock()
        bounding_boxes, _ = detect_face.detect_face(image, minsize,
                                                    self.__pnet, self.__rnet, self.__onet,
                                                    threshold, factor)
        # print(time.clock() - t1)
        nums = len(bounding_boxes)
        img_size = np.asarray(image.shape)[0:2]
        rects = []
        faces = []
        for i in range(nums):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - (det[2] - det[0]) * 0.4, 0)
            bb[1] = np.maximum(det[1] - (det[3] - det[1]) * 0.5, 0)
            bb[2] = np.minimum(det[2] + (det[2] - det[0]) * 0.6, img_size[1])
            bb[3] = np.minimum(det[3] + (det[3] - det[1]) * 0.1, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            rects.append(bb)

            cropped = cv2.resize(cropped, (self.image_size, self.image_size))
            pre_whitened = self.__pre_whiten(cropped)
            faces.append(pre_whitened)
        return rects, faces


def test_face_detect():
    cp = Compare('D:/work/compare/models')
    img = cv2.imread("./1523.jpg")
    for i in range(50):

        rect, face = cp.detect_facerect(img)


if __name__ == '__main__':
    # cp = Compare('D:/work/compare/models')
    # # img_dir = 'D:/work/test_samples/p1/'
    # # neg_dir = 'D:/work/test_samples/n/'
    # # cp.eval_in_class(img_dir)
    # # cp.eval_btw_class(img_dir, neg_dir)
    # data_path = "D:/work/data/shows_dataset_mtcnn_182/"
    # #
    # img1 = cv2.imread(data_path + "fanbingbing" + "/" + "%04d" % 10 + ".png")
    # img1 = cv2.resize(img1, (160, 160))
    # img2 = cv2.imread(data_path + "fanbingbing" + "/" + "%04d" % 11 + ".png")
    # img2 = cv2.resize(img2, (160, 160))
    # #
    # # sss = cp.extract_single_feature(img)
    # # print(type(sss))
    # # print(len(sss))
    # # print(sss)
    # # print(np.shape(sss))
    # # person_list = os.listdir(data_path)
    # #
    #
    # for i in range(50):
    #     t1 = time.clock()
    #     score = cp.eval_dist(img1, img2)
    #     print("time: %s" % (time.clock() - t1))

    test_face_detect()

    # score_data = {}
    # for person in person_list:
    #     score_data[person] = []
    #
    # for per in person_list:
    #     for i in range(500):
    #         img = cv2.imread(data_path + per + "/" + "%04d" % i + ".png")
    #         if img is None:
    #             continue
    #         else:
    #             img = cv2.resize(img, (160, 160))
    #             score = cp.extract_single_feature(img)
    #             score_data[per].append(score[0])
    #     print("%s completed.." % per)
    # myload.save("D:/work/data/show_data.pkl", score_data)
