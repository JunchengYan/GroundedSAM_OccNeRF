import os
import copy

import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict


# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import torch
from huggingface_hub import hf_hub_download

data_path = 'img'
save_path = 'tmp'

TEXT_PROMPT = "Car, Highway, Bus, Truck, Terrain, Vegetation, Sidewalk, Bicycle, Barrier, Pedestrian, Manmade, Motorcycle, Crane, Trailer, Cone, Sky"
BOX_TRESHOLD = 0.20
TEXT_TRESHOLD = 0.20

TEXT_PROMPT1 = "Pedestrian, Bus, Truck"
TEXT_PROMPT2 = "Bicycle, Bicyclist"
TEXT_PROMPT3 = "Motorcycle, Motorcyclist"
TEXT_PROMPT4 = "Barrier, Barricade"
TEXT_PROMPT5 = "Sedan, Highway, Terrain, Tree, Sidewalk, Building, Bridge, Pole, Billboard, Light, Ashbin, Crane, Trailer, Cone, Sky"


dic = {'sedan': 1, 
       'highway': 2, 
       'bus': 3, 
       'truck': 4, 
       'terrain': 5, 
       'tree': 6, 
       'sidewalk': 7,
       'bicycle': 8, # 'bicycle': 8, 'bicyclist': 8,
       'bicyclist': 8, 
       'barrier': 9,
       'barricade': 9, 
       'person': 10,
       'pedestrian': 10,
       'building': 11, # manmade
       'bridge': 11, # manmade
       'pole': 11, # manmade
       'billboard': 11, # manmade
       'light': 11, # manmade
       'ashbin': 11, # manmade
       'motorcycle': 12,
       'motorcyclist': 12,
       'crane': 13, 
       'trailer': 14, 
       'cone': 15, 
       'sky': 16}


### FOR VISUAL ###
colors = np.array(
    [
        [0, 0, 0, 153],
        [0, 150, 245, 153],  # car
        [255, 0, 255, 153], # road
        [255, 255, 0, 153],  # bus
        [160, 32, 240, 153],  # truck
        [150, 240, 80, 153],  # terrain 
        [0, 175, 0, 153],  # vegetation
        [75, 0, 75, 153],  # sidewalk 
        [255, 192, 203, 153],  # bicycle
        [255, 120, 50, 153],  # barrier
        [255, 0, 0, 153],  # pedestrian
        [230, 230, 250, 153],  # manmade
        [200, 180, 0, 153],  # motorcycle
        [0, 255, 255, 153],  # construction_vehicle
        [135, 60, 0, 153],  # trailer 
        [255, 240, 150, 153],  # traffic_cone 
        [102, 255, 255, 153], # sky
        [0, 255, 127, 153],  # ego car
        [111, 111, 222, 153] # others  #light purple
    ]
).astype(np.uint8)

def load_model_hf(device='cpu'):
    cache_config_file = './models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/GroundingDINO_SwinB.cfg.py'
    cache_file = './models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swinb_cogcoor.pth'
    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    
    args.device = device

    checkpoint = torch.load(cache_file, map_location='cpu')
    
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model 


def show_mask(mask, image, phrases, logits, random_color=True, visual=False):
    h, w = mask.shape[-2:]
    img = Image.fromarray(image).convert("RGBA")
    num_mask = mask.shape[0]
    
    logits_masks = torch.zeros((h, w, 1))
    for i in range(num_mask):
        m = mask[i]
        logits_mask = (m.cpu().reshape(h, w, 1) * logits[i].cpu().reshape(1, 1, -1))
        logits_masks = torch.concat([logits_masks, logits_mask], dim=-1)
    
    idx_mask = logits_masks.argmax(dim=-1).numpy()
    phrases.insert(0, 'uncertain')
    catagory_mask = np.vectorize(lambda x: phrases[x])(idx_mask)
    
    vectorized_mapping = np.vectorize(lambda x: dic.get(x, -1))
    norm_idx_mask = vectorized_mapping(catagory_mask).squeeze()
    mask_image = np.zeros((h, w, 4))
    
    # visual:
    
    if visual:
        for i in range(norm_idx_mask.shape[0]):
            for j in range(norm_idx_mask.shape[1]):
                mask_image[i, j] = colors[norm_idx_mask[i, j]]
        mask_image_pil = Image.fromarray((mask_image).astype(np.uint8)).convert("RGBA")
        img = Image.alpha_composite(img, mask_image_pil)
        return np.array(img), norm_idx_mask
    else:
        return norm_idx_mask


def show_mask_sam(mask, image, phrases, logits, random_color=True, visual=False):
    h, w = mask.shape[-2:]
    idx_mask = mask.argmax(dim=0).squeeze().cpu().numpy()
    # from pdb import set_trace; set_trace()
    catagory_mask = np.vectorize(lambda x: phrases[x])(idx_mask)
    
    vectorized_mapping = np.vectorize(lambda x: dic.get(x, -1))
    norm_idx_mask = vectorized_mapping(catagory_mask).squeeze()
    mask_image = np.zeros((h, w, 4))
    
    
    if visual:
        for i in range(norm_idx_mask.shape[0]):
            for j in range(norm_idx_mask.shape[1]):
                mask_image[i, j] = colors[norm_idx_mask[i, j]]
        mask_image_pil = Image.fromarray((mask_image).astype(np.uint8)).convert("RGBA")
        img = Image.fromarray(image).convert("RGBA")
        img = Image.alpha_composite(img, mask_image_pil)
        return np.array(img), norm_idx_mask
    else:
        return norm_idx_mask

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'
    
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    
    sam_checkpoint = './sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    for pic in os.listdir(data_path):
        print("Processing...", pic)
        from time import time
        start = time()
        local_image_path = '{}/{}'.format(data_path, pic)
        image_source, image = load_image(local_image_path)
        

        
        prompt_list = TEXT_PROMPT1.split(",")
        boxes_list, logits_list, phrases_list = [], [], []
        for i in range(len(prompt_list)):
            text = prompt_list[i].strip()
            boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=text, 
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device='cuda')
            
            boxes_list.append(boxes)
            logits_list.append(logits)
            phrases_list += phrases

        prompts = [TEXT_PROMPT2, TEXT_PROMPT3]
        for PROMPT in prompts:
            boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=PROMPT,
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device='cuda')      
            boxes_list.append(boxes)
            logits_list.append(logits)
            phrases_list += phrases

        prompts2 = [TEXT_PROMPT4, TEXT_PROMPT5]
        for PROMPT in prompts2:
            boxes, logits, phrases = predict(
            model=groundingdino_model, 
            image=image, 
            caption=PROMPT,
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device='cuda')
            
            boxes_list.append(boxes)
            logits_list.append(logits)
            phrases_list += phrases


        boxes = torch.cat(boxes_list, dim=0)
        logits = torch.cat(logits_list, dim=0)
        phrases = phrases_list


         
        valid_indices = [index for index in range(len(phrases)) if phrases[index] in dic]
        
        boxes = boxes[valid_indices]
        logits = logits[valid_indices]
        phrases = [phrases[index] for index in valid_indices]
        
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        
        Image.fromarray(annotated_frame).save('{}/{}_annotated.png'.format(data_path, pic[:-4]))
        
        sam_predictor.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        
        return_logits = False
        if return_logits:
            masks, logits_masks, _, _ = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        return_logits=True,
                        multimask_output = False,
                    )
            phrases_sam = copy.deepcopy(phrases)
            frame_with_mask, _ = show_mask(masks, image_source, phrases, logits, visual=True)
            frame_with_mask_sam, _ = show_mask_sam(logits_masks, image_source, phrases_sam, logits, visual=True)
            Image.fromarray(frame_with_mask).save('{}/{}_mask.png'.format(save_path, pic[:-4]))
            Image.fromarray(frame_with_mask_sam).save('{}/{}_mask_sam.png'.format(save_path, pic[:-4]))
        else:
            masks, _, _ = sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        return_logits=False,
                        multimask_output = False,
                    )
            frame_with_mask, norm_idx_mask = show_mask(masks, image_source, phrases, logits, visual=True)
            Image.fromarray(frame_with_mask).save('{}/{}_mask.png'.format(save_path, pic[:-4]))
        print("time: ", time() - start)

if __name__ == "__main__":
    main()