import cv2
import numpy as np
import os
from PIL import Image

def remove_icc_profile(image):
    """รับ numpy array ของภาพแล้วลบ ICC Profile (ไม่บันทึกไฟล์)"""
    try:
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # แปลงเป็น PIL Image
        img = img.convert("RGB")  # ลบ ICC Profile โดยแปลงเป็น RGB
        img_np = np.array(img)  # แปลงกลับเป็น NumPy array
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # แปลงกลับเป็น BGR
    except Exception as e:
        print(f"Error removing ICC profile: {e}")
        return image  # ถ้าลบไม่ได้ ให้ใช้ภาพเดิม

def resize_image(image, target_size):
    """ปรับขนาดภาพให้พอดีกับ target_size โดยรักษาอัตราส่วนและเติมขอบสีดำ"""
    if image is None:
        return None

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # คำนวณ Padding
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))

def process_image(image_path, target_size):
    """โหลดภาพ, ลบ ICC Profile และปรับขนาด"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = remove_icc_profile(image)  # ลบ ICC Profile
    return resize_image(image, target_size)  # ปรับขนาดภาพ

def load_images(images_dir, mask_dir, target_size=512):
    """โหลดภาพจากโฟลเดอร์, ลบ ICC Profile, และปรับขนาด"""
    images = [
        process_image(os.path.join(images_dir, img), target_size) 
        for img in os.listdir(images_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    masks = [
        process_image(os.path.join(mask_dir, mask), target_size) 
        for mask in os.listdir(mask_dir) if mask.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    images = [img for img in images if img is not None]
    masks = [mask for mask in masks if mask is not None]
    
    return images, masks

def main():
    images, masks = load_images('resources/images', 'resources/masks', target_size=256)

    # แสดงภาพด้วย OpenCV
    if images and masks:
        cv2.imwrite('Image 1.png', images[0])
        cv2.imwrite('Mask 1.png', masks[0])
    else:
        print("No valid images or masks found.")

if __name__ == '__main__':
    main()
