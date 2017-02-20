import numpy as np
import cv2


# tutaj src to obraz, z ktorego piksele beda wklejane do obrazu dst
# feather amount to 用来调节边缘模糊程度
def blendImages(src, dst, mask, featherAmount=0.02):
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    # mAX = np.max(maskPts, axis=0)
    # mIN = np.min(maskPts, axis=0)
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)
        pass
        # dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    # Just an alpha blending...
    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
                                                                                                                 1 - weights[
                                                                                                                     :,
                                                                                                                     np.newaxis]) * \
                                                                                                                 dst[
                                                                                                                     maskIndices[
                                                                                                                         0],
                                                                                                                     maskIndices[
                                                                                                                         1]]

    return composedImg


# uwaga, tutaj src to obraz, z ktorego brany bedzie kolor
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst
