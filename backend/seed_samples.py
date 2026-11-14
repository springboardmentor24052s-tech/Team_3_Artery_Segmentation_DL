import cv2
import numpy as np
from skimage.morphology import skeletonize
from ai_engine import VascularAIEngine


def generate_reference_outputs():
    img = np.full((512, 512), 185, dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    vignette = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
    img = np.clip(img - (vignette * 0.28), 35, 220).astype(np.uint8)

    pts_upper = np.array([[100, 70], [160, 130], [215, 210]], np.int32)
    cv2.polylines(img, [pts_upper], False, 30, thickness=18, lineType=cv2.LINE_AA)

    pts_sten = np.array([[215, 210], [235, 245], [250, 280]], np.int32)
    cv2.polylines(img, [pts_sten], False, 32, thickness=5, lineType=cv2.LINE_AA)

    pts_lower = np.array([[250, 280], [310, 365], [395, 450]], np.int32)
    cv2.polylines(img, [pts_lower], False, 30, thickness=16, lineType=cv2.LINE_AA)

    pts_br1 = np.array([[180, 165], [270, 150], [375, 125]], np.int32)
    cv2.polylines(img, [pts_br1], False, 55, thickness=7, lineType=cv2.LINE_AA)

    pts_br2 = np.array([[280, 325], [210, 385], [140, 445]], np.int32)
    cv2.polylines(img, [pts_br2], False, 60, thickness=6, lineType=cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite("sample_angiogram.png", img)

    engine = VascularAIEngine()
    raw_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, enhanced_gray, _ = engine.preprocess_image(raw_bgr)

    base_mask, _ = engine.segment_vessels(enhanced_gray, model_type="baseline")
    base_cl = (skeletonize(base_mask > 0).astype(np.uint8)) * 255
    cv2.imwrite("baseline_mask.png", base_mask)
    cv2.imwrite("baseline_centerline.png", base_cl)

    att_mask, _ = engine.segment_vessels(enhanced_gray, model_type="attention")
    att_cl = (skeletonize(att_mask > 0).astype(np.uint8)) * 255
    cv2.imwrite("attention_mask.png", att_mask)
    cv2.imwrite("attention_centerline.png", att_cl)

    print("Generated sample_angiogram.png, baseline & attention masks/centerlines successfully!")


if __name__ == "__main__":
    generate_reference_outputs()
