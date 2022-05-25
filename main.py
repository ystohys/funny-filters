import sys
from models.simplenet import SimpleNet
from data_utils.preprocessing import *
from filters import mouth_blur


def main(filter):

    trained_nn = SimpleNet()
    trained_nn.load_state_dict(torch.load('pretraineds/trained1.pth',
                                          map_location=torch.device('cpu')))
    face_detect = cv2.CascadeClassifier('pretraineds/haarcascade_frontalface_default.xml')
    captured = cv2.VideoCapture(0)

    while True:
        _, img = captured.read()
        # Conversion to grayscale
        gray_capt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detect.detectMultiScale(gray_capt,
                                             1.1,
                                             4,
                                             minSize=np.array([128, 128]),
                                             maxSize=np.array([512, 512]))

        new_img = img
        # small_gray = gray_capt
        for (x, y, w, h) in faces:
            y += 20
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            small_gray = cv2.resize(gray_capt[y:y+h, x:x+w], dsize=(96, 96))
            tensor_gray = tensorify_img(small_gray)
            tensor_kp = trained_nn(tensor_gray)
            kp = upsize_kp(tensor_kp, x, y, w, h)

            if filter == 'mouthblur':
                new_img = mouth_blur.mouthblur(img, kp)

            for point in kp:
                cv2.circle(new_img, point, 5, (0, 255, 0))

        cv2.imshow('Filtered', new_img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    captured.release()
    cv2.destroyWindow('Filtered')
    cv2.waitKey(1)


if __name__ == '__main__':
    print('SNAPCHAT FILTER ON')
    main(str(sys.argv[1]))
