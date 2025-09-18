import cv2
import numpy as np
import os
from random import randint

spoof_folder = r'C:\Users\mokht.LAPTOP-2TQ72D8U\Desktop\fece Detection_Recognition\SpoofImage'

input_adr_training = r'C:\Users\mokht.LAPTOP-2TQ72D8U\Desktop\fece Detection_Recognition\Training Set' 
input_adr_validation = r'C:\Users\mokht.LAPTOP-2TQ72D8U\Desktop\fece Detection_Recognition\Validation Set' 

output_spoof_adr_training = r'C:\Users\mokht.LAPTOP-2TQ72D8U\Desktop\fece Detection_Recognition\SpoofImage\Training Set'
output_spoof_adr_validation = r'C:\Users\mokht.LAPTOP-2TQ72D8U\Desktop\fece Detection_Recognition\SpoofImage\Validation Set'


# Illumination Transformations

def apply_blur(img, ksize=7):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_color_shift(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shift = randint(-15, 15)
    h = hsv[..., 0].astype(np.int16)
    h = (h + shift) % 180
    hsv[..., 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_replay_artifact(img):
    glare = np.zeros_like(img)
    cv2.circle(glare, (img.shape[1]//2, img.shape[0]//2), min(img.shape[:2])//3, (255,255,255), -1)
    return cv2.addWeighted(img, 0.8, glare, 0.2, 0)

def apply_dark_illumination(img):
    return cv2.convertScaleAbs(img, alpha=0.4, beta=-40)

def apply_backlight(img):
    rows, cols, _ = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.rectangle(mask, (cols//2, 0), (cols, rows), (255, 255, 255), -1)
    return cv2.addWeighted(img, 1.0, mask, 0.5, 0)

# Angle Simulation

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# Shape Simulation

def warp_inside(img):
    h, w = img.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_x += 15.0 * np.sin(2 * np.pi * map_y / 150)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def warp_outside(img):
    h, w = img.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_x -= 15.0 * np.sin(2 * np.pi * map_y / 150)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def warp_corner(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([[0, 0], [w, 0], [w * 0.2, h], [w * 0.8, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h))

# Paper Cut Spoof 

def simulate_paper_cut_spoof(img, cut='face'):
    mask = np.ones_like(img) * 255
    h, w = img.shape[:2]
    if cut == 'face':
        cv2.circle(mask, (w//2, h//2), min(h, w)//4, (0, 0, 0), -1)
    elif cut == 'upper':
        mask[h//4:h//2, :] = 0
    return cv2.bitwise_and(img, mask)


transformations = [
    ("blur", lambda img: apply_blur(img)),
    ("color_shift", lambda img: apply_color_shift(img)),
    ("replay_artifact", lambda img: apply_replay_artifact(img)),
    ("dark", lambda img: apply_dark_illumination(img)),
    ("backlight", lambda img: apply_backlight(img)),
    ("rotate_p15", lambda img: rotate_image(img, 15)),
    ("rotate_m15", lambda img: rotate_image(img, -15)),
    ("warp_inside", lambda img: warp_inside(img)),
    ("warp_outside", lambda img: warp_outside(img)),
    ("warp_corner", lambda img: warp_corner(img)),
    ("paper_face", lambda img: simulate_paper_cut_spoof(img, 'face')),
    ("paper_upper", lambda img: simulate_paper_cut_spoof(img, 'upper'))
]


def apply_spoof_to_folder(input_path, output_path):
    for person_name in os.listdir(input_path):
        person_folder = os.path.join(input_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        output_person_folder = os.path.join(output_path, person_name)
        os.makedirs(output_person_folder, exist_ok=True)

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f" Failed to read image: {img_path}")
                continue

            base_name, ext = os.path.splitext(img_name)
            ext = ext.lower()
            orig_size_kb = os.path.getsize(img_path) / 1024

            for trans_name, trans_func in transformations:
                try:
                    transformed = trans_func(img)
                    transformed = cv2.resize(transformed, (img.shape[1], img.shape[0]))

                    out_name = f"{base_name}_{trans_name}{ext}"
                    out_path = os.path.join(output_person_folder, out_name)

                    
                    if ext in ['.jpg', '.jpeg']:
                        quality = 95
                        for _ in range(10):
                            cv2.imwrite(out_path, transformed, [cv2.IMWRITE_JPEG_QUALITY, quality])
                            spoof_size_kb = os.path.getsize(out_path) / 1024
                            if 0.8 * orig_size_kb <= spoof_size_kb <= 1.2 * orig_size_kb:
                                break
                            if spoof_size_kb > 1.2 * orig_size_kb:
                                quality -= 10
                            elif spoof_size_kb < 0.8 * orig_size_kb:
                                quality = min(quality + 5, 100)
                    else:
                        cv2.imwrite(out_path, transformed)

                except Exception as e:
                    print(f" Error in {trans_name} for {img_name}: {e}")


apply_spoof_to_folder(input_adr_training, output_spoof_adr_training)
apply_spoof_to_folder(input_adr_validation, output_spoof_adr_validation)