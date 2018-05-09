import numpy as np
import dlib
import cv2
from os import listdir
from os.path import isfile, isdir, join
from face_helper import rect_to_bb, shape_to_np, FACIAL_LANDMARKS_IDXS
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '/Users/kirillovchinnikov/.virtualenvs/chat_env/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

EMO_DICT = {
    -1: 'Not file',
    0: 'Neutral',
    1: 'Anger',
    2: 'Contempt',
    3: 'Disgust',
    4: 'Fear',
    5: 'Happy',
    6: 'Sad',
    7: 'Surprise'
}


def shape_rotate(shape, angle, dtype="float"):
    coords = np.zeros((len(shape), 2), dtype=dtype)
    angle = angle * np.pi / 180.0
    for i in range(0, len(shape)):
        coords[i][0] = shape[i][0] * np.cos(angle) + shape[i][1] * np.sin(angle)
        coords[i][1] = -shape[i][0] * np.sin(angle) + shape[i][1] * np.cos(angle)

        # print(coords[i][0] - shape[i][0])
        # print(coords[i][1] - shape[i][1])

    return coords


def shape_normalize(shape):
    coords = np.zeros((len(shape), 2), dtype='float')

    maxX = np.amax(shape[:, 0])
    minX = np.amin(shape[:, 0])
    maxY = np.amax(shape[:, 1])
    minY = np.amin(shape[:, 1])

    scaleX = 1 / (maxX - minX)
    scaleY = 1 / (maxY - minY)

    for i in range(0, len(shape)):
        coords[i][0] = (shape[i][0] - minX) * scaleX
        coords[i][1] = (shape[i][1] - minY) * scaleY

    return coords


kanade_set = '/Users/kirillovchinnikov/Downloads/binary/cohn-kanade-images'

kanade_emotions_set = '/Users/kirillovchinnikov/Downloads/binary/Emotion'

dir_list = [f for f in listdir(kanade_set) if isdir(join(kanade_set, f))]

data_for_X = []

data_for_y = []

for dir_kanade in dir_list:
    subdir_folder = join(kanade_set, dir_kanade)
    subdir_list = [f for f in listdir(subdir_folder) if isdir(join(subdir_folder, f))]
    for image_subdir in subdir_list:
        image_folder = join(subdir_folder, image_subdir)
        target_emotion_folder = join(join(kanade_emotions_set, dir_kanade), image_subdir)
        target_emotion = -1
        print(target_emotion_folder)
        if isdir(target_emotion_folder):
            target_emotion_files = [f for f in listdir(target_emotion_folder)
                                    if isfile(join(target_emotion_folder, f))
                                    and f.split('.')[-1] == 'txt']
            print(target_emotion_files)
            if len(target_emotion_files) > 0:
                target_emotion_file = join(target_emotion_folder, target_emotion_files[0])
                with open(target_emotion_file, 'r') as emotion_file:
                    target_emotion = int(float(emotion_file.read()))
        image_list = [f for f in listdir(image_folder) if isfile(join(image_folder, f)) and f.split('.')[-1] == 'png']
        image_list.sort()
        images_len = len(image_list)
        for i, image_filename in enumerate(image_list):
            image_emotion = -1
            # if i <= 2 and i <= (images_len / 2) and target_emotion != -1:
            #     image_emotion = 0
            if i > 3 and i > (images_len / 2) and target_emotion != -1:
                image_emotion = target_emotion
            else:
                continue


            image_file = join(image_folder, image_filename)
            image = cv2.imread(image_file, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            if len(rects) > 0:
                rect = rects[0]
            else:
                continue

            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]

            leftEyeCenter = leftEyePts.mean(axis=0)
            rightEyeCenter = rightEyePts.mean(axis=0)

            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180.0

            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            shape_r = shape_rotate(shape, angle)
            shape_n = shape_normalize(shape_r)

            data_for_X.append(shape_n)
            data_for_y.append(image_emotion)


            shape_size = 100
            shape_image = np.zeros((shape_size, shape_size), dtype='int8')
            for (x, y) in shape_n:
                cv2.circle(shape_image, (int(x * shape_size), int(y * shape_size)), 1, (255, 255, 255), -1)

            cv2.putText(image, "{}".format(EMO_DICT[image_emotion]), (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Output", shape_image)
            cv2.imshow("Output1", image)
            cv2.waitKey(0)
