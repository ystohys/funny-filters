import cv2


def mouthblur(image, keypoints, pad=20):
    mouth_kp = keypoints[-1:-5:-1]
    mouth_btm, mouth_top, mouth_right, mouth_left = mouth_kp
    mouth_box_width = mouth_right[0] - mouth_left[0]
    mouth_box_height = mouth_btm[1] - mouth_top[1]
    y = mouth_top[1]
    x = mouth_left[0]
    mouth_box = image[y-pad:y+mouth_box_height+pad,
                      x-pad:x+mouth_box_width+pad]
    if mouth_box.size > 0:
        blur_box = cv2.GaussianBlur(mouth_box, (35, 35), 100)
        image[y-pad:y+mouth_box_height+pad, x-pad:x+mouth_box_width+pad] = blur_box

    return image

