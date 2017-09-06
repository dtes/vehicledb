import cv2 as cv
import numpy as np

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

showDetails = False
showImages = False
offset = 0
save_path = 'static/images/00-lp.jpg'


# ##############################################
# etot metod nahodit vo vhodnoi kartinke oblast' gde predpolozhitel'no est nomer
def detect_lp(name):
    lp_cascade = cv.CascadeClassifier('data/haarcascade_russian_plate_number.xml')

    img = cv.imread(name)
    copy = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    lp = None
    plates = lp_cascade.detectMultiScale(gray, 1.15, 6)
    for (x, y, w, h) in plates:
        lp = copy[y:y + h, x:x + w]
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 150), 2)
        # if showDetails: cv.imwrite('detected-lp.jpg', copy[y:y + h, x:x + w])

    if showImages:
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return lp


# ##############################################
# additional method
def containsButNotEquals(rect1, rect2):
    x, y, w, h = rect1
    x2, y2, w2, h2 = rect2

    if rect1 == rect2:
        return False

    if x <= x2 and y <= y2 and x + w >= x2 + w2 and y + h >= y2 + h2:
        return True
    return False


# ##############################################
# etot metod iz naidennoi oblasti ishet chisla, bukvy
def charDetect(lp):
    char_contours = []
    # region find all contours in license plate
    img = lp.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # thr = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # blur = cv.fastNlMeansDenoising(thr)
    thr = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)
    H = img.shape[0]
    W = img.shape[1]

    if showImages:
        cv.imshow('thr', thr)
        cv.waitKey(0)

    img1, contours, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda cnt: cv.boundingRect(cnt)[0])
    # endregion

    # region detecting contours which may be characters
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        boundRectArea = w * h

        # if w < W * .6 and h > H * .2 and w / h < 1.5 and h / w < 3:
        if w < W * .6 and h > H * .2 and w / h < 1.5 and h / w < 3 and area / boundRectArea > .1:
            # print cv.minAreaRect(contour)
            char_contours.append(contour)
    # endregion

    # region for contours like a 0 and D we remove inner contours
    contoursIdxToRemove = []
    for i, contour in enumerate(char_contours):
        for j, contour2 in enumerate(char_contours):
            rect1 = cv.boundingRect(contour)
            rect2 = cv.boundingRect(contour2)
            if containsButNotEquals(rect1, rect2):
                x, y, w, h = rect1
                x2, y2, w2, h2 = rect2
                ratio = float(w2 * h2) / float(w * h)
                # print w * h, w2 * h2, ratio

                if ratio > .4:
                    contoursIdxToRemove.append(j)
                    if showDetails:
                        cv.imshow('rect1', img[y:y + h, x:x + w])
                        cv.imshow('rect2', img[y2:y2 + h2, x2:x2 + w2])
                        cv.waitKey(0)

    contoursIdxToRemove.reverse()
    for idx in contoursIdxToRemove:
        del char_contours[idx]
    # endregion

    # region remove contours which very near to border
    contoursIdxToRemove = []
    for i, contour in enumerate(char_contours):
        x, y, w, h = cv.boundingRect(contour)
        if y < 5 or H - y - h < 5 or x < 5 or W - x - w < 5:
            contoursIdxToRemove.append(i)

    contoursIdxToRemove.reverse()
    for idx in contoursIdxToRemove:
        del char_contours[idx]
    # endregion

    # region remove right rectangle which contains two chars
    count_inner_contours = [0 for i in range(len(char_contours))]
    for i, contour in enumerate(char_contours):
        for contour2 in char_contours:
            r1 = cv.boundingRect(contour)
            r2 = cv.boundingRect(contour2)
            if containsButNotEquals(r1, r2):
                count_inner_contours[i] += 1

    count_inner_contours.reverse()
    for i, count in enumerate(count_inner_contours):
        if count > 1:
            print ''
            del char_contours[len(char_contours) - i - 1]
    # endregion

    # region symbols like a 0 and D
    for i, contour in enumerate(char_contours):
        for j, contour2 in enumerate(char_contours):
            r1 = cv.boundingRect(contour)
            r2 = cv.boundingRect(contour2)
            if containsButNotEquals(r1, r2):
                del char_contours[j]

    # endregion

    # region draw bounding rectangle around contours
    for contour in char_contours:
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        boundRectArea = w * h

        # print 'width, height: ', str(w), str(h)
        # print 'area', area, 'boundRectArea', w * h

        cv.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 0, 255), 1)
        # cv.drawContours(img, [contour], -1, (0, 0, 220), 1)
        if showImages:
            cv.imshow('contours', img)
            # if writeChars:
            #     cv.imwrite('data/chars/' + str(x) + str(y) + image_name, lp[y:y + h, x:x + w])

    cv.imwrite(save_path, img)
    # endregion

    return char_contours


# ##############################################
# zdes' idet raspoznovanie simvolov naidennih v predidushem metode
def charRecognize(lp, char_contours):
    npaClassifications = np.loadtxt("data/classifications.txt", np.float32)  # read in training classifications
    npaFlattenedImages = np.loadtxt("data/flattened_images.txt", np.float32)  # read in training images

    kNearest = cv.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv.ml.ROW_SAMPLE, npaClassifications)

    gray = cv.cvtColor(lp.copy(), cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    thr = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 2)

    lp_number = ''
    for contour in char_contours:
        x, y, w, h = cv.boundingRect(contour)
        imgROI = thr[y - offset: y + h + offset, x - offset: x + w + offset]

        if showDetails:
            cv.imshow('imgROI', imgROI)
            cv.waitKey(0)

        imgROIResized = cv.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        lp_number += strCurrentChar

    return lp_number


# glavnyi metod kotoryi vyzyvaet vse vishe opisannye metody
def get_lp_number(name):
    lp = detect_lp(name)

    if lp is None:
        return 'not recognized'

    char_contours = charDetect(lp)
    lp_number = charRecognize(lp, char_contours)
    return lp_number
