import cv2
import numpy as np
import time


# image = cv2.imread('test.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imwrite('gray.jpg', gray)

def take_pic():
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    cap.release()
    return frame


def crop_face(img):

    cascadeFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = cascadeFace.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    (x, y, h, w) = faces[0]

    cropped_img = img[y:y + h, x:x + w]
    # cv2.imwrite('cropped.jpg', cropped_img)
    return cropped_img


def compare_faces(img):
    base = cv2.imread('base.jpg')
    # test1 = cv2.imread('test1.jpg')
    # test2 = cv2.imread('test2.jpg')
    basehsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hrange = [0, 180]
    srange = [0, 256]
    ranges = hrange + srange

    histbase = cv2.calcHist(basehsv, [0, 1], None, [180, 256], ranges)
    cv2.normalize(histbase, histbase, 0, 255, cv2.NORM_MINMAX)

    histimg = cv2.calcHist(imghsv, [0, 1], None, [180, 256], ranges)
    cv2.normalize(histimg, histimg, 0, 255, cv2.NORM_MINMAX)

    base_test = cv2.compareHist(histbase, histimg, 0)

    return base_test
