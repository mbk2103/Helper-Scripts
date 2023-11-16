import json
import os
from pycocotools import mask
from PIL import Image, ImageDraw

json_path= r"C:\Users\PC_4236\Desktop\ytvis-dataset-class-car\train\JPEGImages\mclaren-train\_annotations.coco.json"

output_folder = r"C:\Users\PC_4236\Desktop\ytvis-dataset-class-car\train\JPEGImages\mclaren-train\Annotations"
os.makedirs(output_folder, exist_ok=True)

with open(json_path, 'r') as f:
    annotations_data = json.load(f)

annotations = annotations_data["annotations"]
images = annotations_data["images"]

for annotation in annotations:
    area = annotation["area"]
    bbox = annotation["bbox"]
    category_id = annotation["category_id"]
    annotation_id = annotation["id"]
    image_id = annotation["image_id"]
    is_crowd = annotation["iscrowd"]
    segmentation = annotation["segmentation"]
    
    image_info = next(image_info for image_info in images if image_info["id"] == image_id)
    file_name = image_info["file_name"]
    # Create Mask
    rle = mask.frPyObjects(segmentation, annotation["bbox"][3], annotation["bbox"][2])
    mask_array = mask.decode(rle)
    mask_image = Image.fromarray(mask_array[:, :, 0] * 255).convert("L")

    mask_filename = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".png")
    
    mask_image.save(mask_filename)

    print(f"Annotation ID: {annotation_id}, Image ID: {image_id}, Category ID: {category_id}")
    print(f"Mask saved: {mask_filename}")