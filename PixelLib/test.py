# pip3 install pycocotools
# pip3 install pixellib
# pip3 install pixellib --upgrade
#-----------------------------------------------------
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import torch
import torchvision

#Must version of torch == version of torchvision  == cu113
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ins = instanceSegmentation()
# Download the model (resnet50) (https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl)
# model              (resnet101)(https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet101.pkl)

ins.load_model("pointrend_resnet50.pkl", confidence = 0.3,detection_speed = "rapid")
#detection_speed = "rapid ## The rapid mode achieves 0.15 seconds for processing a single image.
#detection_speed = "fast ## The fast mode achieves 0.20 seconds for processing a single image.

ins.segmentImage("image.jpeg", show_bboxes=True, output_image_name="output_image_box.jpg")
#---------------------------------------------------------------------------------------
results, output = ins.segmentImage("image.jpeg", show_bboxes=True, output_image_name="result.jpg")
print(results["class_ids"])

#Custom Object Detection in Image Segmentation
#The PointRend model used is a pretrained COCO model which supports 80 classes of objects
#-----------------------------------------------------------------------------------------
#We want to filter the detections of our sample image to detect only person in the image.

target_classes = ins.select_target_classes(person = True)
results2, output2=ins.segmentImage("image.jpeg", segment_target_classes = target_classes, output_image_name="person.jpg")
print(results2["class_ids"])
#------------------------------------------------------------------------------------------
#Object Extractions in Images
ins.segmentImage("image.jpeg", show_bboxes=True, extract_segmented_objects=True,
save_extracted_objects=True, output_image_name="Extractions.jpg" )

#Extraction from Bounding Box Coordinates
ins.segmentImage("image.jpeg", extract_segmented_objects = True, extract_from_box = True, save_extracted_objects = True, output_image_name="extract_from_box.jpg")

# Modifications for Better Visualization
ins.segmentImage("image.jpeg", show_bboxes=True, text_size=0.5, text_thickness=1, box_thickness=10, output_image_name="Better_Visualization.jpg")

hh,j=ins.segmentBatch("input", show_bboxes=True, extract_segmented_objects=True,
save_extracted_objects=True, output_folder_name="output")
print(hh)