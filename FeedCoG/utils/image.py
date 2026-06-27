from PIL import Image, ImageDraw
import numpy as np

def compute_crop_box(box_list, img_size, ratio: float = 0.20):
    # (x1, y1), (x2, y2)
    if not box_list:
        # 如果没有有效的box，返回整个图像
        w, h = img_size
        return (0, 0, w, h)
    
    # 过滤掉无效的box
    valid_boxes = [box for box in box_list if box is not None and len(box) == 4]
    if not valid_boxes:
        w, h = img_size
        return (0, 0, w, h)
    
    x1_list = [item[0] for item in valid_boxes]
    y1_list = [item[1] for item in valid_boxes]
    x2_list = [item[2] for item in valid_boxes]
    y2_list = [item[3] for item in valid_boxes]

    # compute the crop-box
    w, h = img_size
    x1 = min(x1_list) - w * ratio
    y1 = min(y1_list) - h * ratio
    x2 = max(x2_list) + w * ratio
    y2 = max(y2_list) + h * ratio
    
    # validation check
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # 确保坐标有效
    if x1 >= x2:
        x1 = 0
        x2 = w
    if y1 >= y2:
        y1 = 0
        y2 = h
    
    return (int(x1), int(y1), int(x2), int(y2))

def mask_image(input_image, mask_coord, mode="random"):
    """
    Args:
        input_image: PIL.Image
        mask_coord: [(x1,y1), (x2,y2)]
        mode: mean or black
    """
    #  clone a copy of img_rgb
    img_rgb_clone = input_image.copy()
    #  generate a mask image
    mask_img = Image.new('L', input_image.size, 0)
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle(mask_coord, fill=255)
    if mode == "mean":
        #  cover the pixels in the mask with mean rgb value of the mask area
        mask_array = np.array(mask_img)
        img_rgb_array = np.array(img_rgb_clone)
        img_rgb_array[mask_array == 255] = np.mean(img_rgb_array[mask_array == 255], axis=0).astype(int)
        img_rgb_clone = Image.fromarray(img_rgb_array)
    elif mode == "black":
        #  cover the pixels in the mask with black
        mask_array = np.array(mask_img)
        img_rgb_array = np.array(img_rgb_clone)
        img_rgb_array[mask_array == 255] = 0
        img_rgb_clone = Image.fromarray(img_rgb_array)
    elif mode == "random":
        #  cover the pixels in the mask with random color
        mask_array = np.array(mask_img)
        img_rgb_array = np.array(img_rgb_clone)
        img_rgb_array[mask_array == 255] = np.random.randint(0, 255, size=img_rgb_array[mask_array == 255].shape)
        img_rgb_clone = Image.fromarray(img_rgb_array)
    return img_rgb_clone