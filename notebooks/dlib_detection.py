# import the necessary packages


import numpy as np
import dlib
import cv2
from catboost import CatBoost

from face_helper import rect_to_bb, shape_to_np, FACIAL_LANDMARKS_IDXS

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/kirillovchinnikov/.virtualenvs/chat_env/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

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


def shape_rotate(shape, dtype="float"):
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    leftEyeCenter = leftEyePts.mean(axis=0)
    rightEyeCenter = rightEyePts.mean(axis=0)

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180.0
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

    scaleMin = scaleY

    if scaleX < scaleY:
        scaleMin = scaleX

    for i in range(0, len(shape)):
        coords[i][0] = (shape[i][0] - minX) * scaleMin
        coords[i][1] = (shape[i][1] - minY) * scaleMin

    return coords

cap = cv2.VideoCapture(0)

from catboost import CatBoostClassifier
model = CatBoostClassifier(loss_function='MultiClass')
model.load_model(fname='model_yan.cbm')

while True:
    ret, frame = cap.read()

    image = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        shape_r = shape_rotate(shape)

        shape_n = shape_normalize(shape_r)

        X_input = []

        X_input.append(np.hstack(shape_n))

        X_input = np.array(X_input)
        X_input.shape
        proba_predict = model.predict_proba(X_input)
        predict = model.predict(X_input)
        num_predict = proba_predict.argmax(axis=1)[0]
        # print(proba_predict,predict, proba_predict.argmax(axis=1)[0])
        emotion = EMO_DICT[num_predict] + ' probab = {}%'.format(proba_predict[0][num_predict])

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Emotion {}".format(emotion), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()