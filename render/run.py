import os
import cv2
import numpy as np
import render.ImageProcessing
import shutil
import render.fasterobj
import pickle
from collections import deque

ROOT = './dataset'
BLEND = ROOT + '/blend'
MASK_OUT = ROOT + '/mask_out'

def bb_from_img(img):
    '''Get the boundingbox of a mask image
    :param img: ndarray
    :return: tuple(),tuple(),tuple()
    '''
    maskIndices = np.where(img != 0)
    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    lefttop = np.max(maskPts, axis=0)
    rightbottom = np.min(maskPts, axis=0)
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    return lefttop,rightbottom,faceSize

ratios = []
# crt_ratios = pickle.load(open('newRatios.pickle', "rb"))
class Draw():
    gaussian = np.array([7, 26, 41, 26, 7])
    rect_count = 0
    crt_ratios = deque([])
    buffer_bgOrgImg = deque([])
    buffer_alignedFace = deque([])
    buffer_final_area = deque([])
    buffer_bgFillImg = deque([])

    def draw_single(self, heroImg,bgOrgImg,bgFillImg,bgmask,clothmask, name, index, frame_num):
        '''
            Align the heroImg by 68 face features to fit the area.
            :param heroImg: The capture of the OpenGL surface to display a 3d model.
            :param bgOrgImg: The original movie frame.
            :param bgFillImg: The real background image to paint on.
            :param bgmask:
            :param name: The filename
            :return:
            '''
        cv2.imshow('heromsk', heroImg)
        cv2.imshow('bgmask', bgmask)
        blend_name = BLEND + '/' + name
        mask_out = MASK_OUT + '/' + name

        rect_hero,bg_rect_center = self.rectLocate(heroImg, bgmask)
        #hero rect--------
        half_h = int(rect_hero.shape[0]/2)
        half_w = int(rect_hero.shape[1]/2)
        #hero rect--------

        alignedFace = np.zeros_like(bgmask)
        alignedFace,rect_hero = self.rectPaste(alignedFace,rect_hero, bg_rect_center,half_w,half_h)

        # Get hero mask
        heromask = np.zeros_like(bgmask)
        heromask[(alignedFace[:, :] != [0, 0, 0]).any(axis=2)] = 255
        cv2.imshow('heromask',heromask)

        intersection = np.bitwise_and(heromask,bgmask)

        union = np.bitwise_or(heromask,bgmask)
        cv2.imwrite(mask_out, union)

        narrows = np.bitwise_xor(heromask, union)
        narrows = np.bitwise_and(narrows, union)

        final_area = union - narrows

        ratio = np.sum(intersection)/np.sum(union)
        ratios.append(ratio)
        self.rect_count += 1

        self.crt_ratios.append(ratio)
        self.buffer_alignedFace.append(alignedFace)
        self.buffer_bgOrgImg.append(bgOrgImg)
        self.buffer_final_area.append(final_area)

        if index <= 4:
            return
        elif index == frame_num-1:
            for i in range(5):
                bgOrgImg = self.buffer_bgOrgImg.popleft()
                alignedFace = self.buffer_alignedFace.popleft()
                final_area = self.buffer_final_area.popleft()
                bgFillImg = self.buffer_bgFillImg.popleft()
                self.final_draw(self, bgOrgImg, alignedFace, final_area, bgFillImg, blend_name)
        else:
            bgOrgImg = self.buffer_bgOrgImg.popleft()
            alignedFace = self.buffer_alignedFace.popleft()
            final_area = self.buffer_final_area.popleft()
            bgFillImg = self.buffer_bgFillImg.popleft()

            correct_r = np.dot(self.gaussian,self.crt_ratios[(- 2):( 3)])/107
            scale_factor = np.sqrt(float(correct_r)/float(ratio))
            alignedFace = self.scaleAdjust(alignedFace, scale_factor,rect_hero,bg_rect_center)

            self.final_draw(self, bgOrgImg, alignedFace, final_area, bgFillImg, blend_name)
            # renderedImg = render.ImageProcessing.colorTransfer(bgOrgImg, alignedFace, final_area)
            #
            # if renderedImg is None:
            #     print('renderedImg is None')
            # cameraImg = render.ImageProcessing.blendImages(renderedImg, bgFillImg, final_area)
            #
            # cv2.imshow('renderedImg', renderedImg)
            # cv2.imshow('out', cameraImg)
            # cv2.imshow('src', bgOrgImg)
            # cv2.imwrite(blend_name, cameraImg)

    def rectPaste(self,alignedFace,rect_hero, bg_rect_center,half_w,half_h):
        self.x = int(bg_rect_center[0] + half_w) - int(bg_rect_center[0] - half_w)
        self.y = int(bg_rect_center[1] + half_h) - int(bg_rect_center[1] - half_h) + 1
        rect_hero = cv2.resize(rect_hero, (self.x, self.y))

        alignedFace[int(bg_rect_center[1] - half_h):int(bg_rect_center[1] + half_h) + 1,
        int(bg_rect_center[0] - half_w):int(bg_rect_center[0] + half_w), :] = rect_hero
        return alignedFace,rect_hero

    def scaleAdjust(self,alignedFace, scale_factor,rect_hero,bg_rect_center):
        '''
        To adjust the scale of head object in alignedFace.
        :param alignedFace:
        :param scale_factor:
        :param rect_hero:
        :param bg_rect_center:
        :return:
        '''
        rect_hero = cv2.resize(rect_hero, (int(self.x * scale_factor), int(self.y * scale_factor)))

        half_h = int(rect_hero.shape[0] / 2)
        half_w = int(rect_hero.shape[1] / 2)
        x = int(2 * half_w)
        y = int(2 * half_h) + 1
        rect_hero = cv2.resize(rect_hero, (x, y))

        temp = alignedFace.copy()
        temp[int(bg_rect_center[1] - half_h):int(bg_rect_center[1] + half_h) + 1,
        int(bg_rect_center[0] - half_w):int(bg_rect_center[0] + half_w), :] = rect_hero
        return temp

    def rectLocate(self,heroImg,bgmask):
        rb_hero, lt_hero, herobb = bb_from_img(heroImg)
        rb_bg, lt_bg, bgbb = bb_from_img(bgmask)
        ratio = np.sqrt(float(bgbb[0] * bgbb[1]) / float(herobb[0] * herobb[1]))  # Edge to adjust

        rect_hero = heroImg[lt_hero[1]:rb_hero[1], lt_hero[0]:rb_hero[0]]
        # Resize to fit the background rect.
        rect_hero = cv2.resize(rect_hero, (int(rect_hero.shape[1] * ratio), int(rect_hero.shape[0] * ratio)))

        bg_rect_center = (rb_bg + lt_bg) / 2
        return rect_hero,bg_rect_center

    def test_clothes(self,cameraImg,bgOrgImg,clothmask):
        clothsno = np.zeros_like(bgOrgImg)
        clothsno[(clothmask[:, :] == [0, 0, 0]).any(axis=2)] = 255
        clothsno[(clothmask[:, :] == [255, 255, 255]).any(axis=2)] = 0
        cv2.imshow('cloth', clothsno)
        cloths = cv2.bitwise_and(bgOrgImg, clothmask)
        nocloths = cv2.bitwise_and(cameraImg, clothsno)
        cv2.imshow('result', nocloths + cloths)

    def final_draw(self,bgOrgImg, alignedFace, final_area, bgFillImg,blend_name):
        renderedImg = render.ImageProcessing.colorTransfer(bgOrgImg, alignedFace, final_area)

        if renderedImg is None:
            print('renderedImg is None')
        cameraImg = render.ImageProcessing.blendImages(renderedImg, bgFillImg, final_area)

        cv2.imshow('renderedImg', renderedImg)
        cv2.imshow('out', cameraImg)
        cv2.imshow('src', bgOrgImg)
        cv2.imwrite(blend_name, cameraImg)

def Go():
    SRC = ROOT + '/ori'
    MASK = ROOT + '/mask'
    INPAINTED = ROOT + '/inpainted'
    # BACKGROUND = ROOT + '/background'
    CLOTHES_MASK = ROOT + '/clothes_mask'
    # rmVerbose(ROOT+'/mask',ROOT+'/the_master') # To balance files number of each dirs.
    if os.path.exists(BLEND):
        shutil.rmtree(BLEND)
        os.mkdir(BLEND)
    else:
        os.mkdir(BLEND)

    if os.path.exists(MASK_OUT):
        shutil.rmtree(MASK_OUT)
        os.mkdir(MASK_OUT)
    else:
        os.mkdir(MASK_OUT)

    # file_names = [int(name.split('.')[0]) for name in os.listdir(SRC)]
    # file_names.sort()
    # file_names = [str(name) + '.jpg' for name in file_names]
    file_names = os.listdir(SRC)

    draw = Draw()
    Model = render.fasterobj.Render()
    for index,name in enumerate(file_names):
        bgOrgImg = cv2.imread(SRC + '/' + name)
        bgFillImg = cv2.imread(INPAINTED + '/' + name)
        bgmask = cv2.imread(MASK + '/' + name)
        heroImg = Model.draw(bgOrgImg)
        clothmask = cv2.imread(CLOTHES_MASK + '/' + name)
        draw.draw_single(heroImg, bgOrgImg, bgFillImg, bgmask, clothmask, name, index, len(file_names))
        pass

    # pickle.dump(Model.Poses, open("Poses.pickle", "wb"))
    # pickle.dump(ratios, open("ratios.pickle", "wb"))

    pass

