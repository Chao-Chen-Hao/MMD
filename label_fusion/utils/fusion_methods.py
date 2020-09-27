import numpy as np
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

def certainty_aggregation(priority_list, label_list, certainty_list, threshold, gpu_id):
    num_classes = len(priority_list)
    img_size = label_list[0].shape # (B, H, W) -> (4, 1024, 2048)
    cubic_size = certainty_list[0].shape # (B, H, W, C) -> (4, 1024, 2048, 19)
    #print(img_size, cubic_size)
    # Generate the pseudo labels
    pseudo_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_mask = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_cubic = torch.zeros(cubic_size).cuda(gpu_id)
    pseudo_label_cubic_sum = torch.zeros(cubic_size).cuda(gpu_id)
    pseudo_label_cubic_count = torch.zeros(cubic_size).cuda(gpu_id)

    # Extract the tensor
    for i in range(num_classes): # c-1 to 0
        # Build the mask of the cubic
        for l in range(len(label_list)): 
            class_map = torch.zeros(img_size).cuda(gpu_id)
            class_map[label_list[l] == i] = 1
            certainty_list[l][:,:,:,i] = certainty_list[l][:,:,:,i]*class_map
            pseudo_label_cubic_count[:,:,:,i] += class_map

    pseudo_label_cubic_count_re = torch.zeros(cubic_size).cuda(gpu_id)
    pseudo_label_cubic_count_re[pseudo_label_cubic_count == 0] = 1
    pseudo_label_cubic_count = pseudo_label_cubic_count + pseudo_label_cubic_count_re

    for l in range(len(label_list)):
        pseudo_label_cubic_sum = pseudo_label_cubic_sum + certainty_list[l]

    pseudo_label_cubic_sum = pseudo_label_cubic_sum / pseudo_label_cubic_count

    # mask those pixels with certainty <90
    pseudo_label_value, pseudo_label_idx = torch.max(pseudo_label_cubic_sum, dim=3)
    pseudo_label_idx.type(torch.uint8)

    pseudo_label_mask[pseudo_label_value >= threshold] = 1
    pseudo_label = pseudo_label_idx*pseudo_label_mask + pseudo_label*(1-pseudo_label_mask)

    del pseudo_label_mask, pseudo_label_cubic, pseudo_label_cubic_sum, pseudo_label_cubic_count
    
    return pseudo_label

def priority_aggregation(priority_list, label_list, certainty_list, threshold, gpu_id):
    num_classes = len(priority_list)
    img_size = label_list[0].shape # (B, H, W) -> (4, 512, 1024)
    #print(img_size)
    # Generate the pseudo labels
    pseudo_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    # Extract the tensor
    for p in range(num_classes-1, -1, -1): # c-1 to 0
        i, m = priority_list[p][0], priority_list[p][1]
        mask_ori = torch.zeros(label_list[m].shape).cuda(gpu_id)
        mask_certainty = torch.zeros(label_list[m].shape).cuda(gpu_id)
        mask_ori[label_list[m] == i] = 1
        mask_certainty[certainty_list[m][:,:,:,i] >= threshold] = 1
        pseudo_label[mask_ori*mask_certainty == 1] = i

    return pseudo_label

def majority_aggregation(priority_list, label_list, certainty_list, threshold, gpu_id, RF=(5, 5)):
    num_classes = len(priority_list)
    img_size = label_list[0].shape
    cubic_size = certainty_list[0].shape # (B, H, W, C) -> (4, 1024, 2048, 19)
    #print(img_size, cubic_size)
    # Generate the pseudo labels
    pseudo_label = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_cubic = torch.zeros((img_size[0],19,img_size[1],img_size[2])).cuda(gpu_id)
    pseudo_label_cubic_mask = torch.zeros((img_size[0],19,img_size[1],img_size[2])).cuda(gpu_id)
    pseudo_label_cubic_certainty = torch.zeros((img_size[0],19,img_size[1],img_size[2])).cuda(gpu_id)

    count_map = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    mask_ori = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    mask_certainty = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    
    # Extract the tensor
    for p in range(num_classes-1, -1, -1): # c-1 to 0
        # assign default label
        i, m = priority_list[p][0], priority_list[p][1]
        pseudo_label[label_list[m] == i] = i
        count_map[label_list[m] == i] += 1

        # Build the cubic
        class_map = torch.zeros(img_size).cuda(gpu_id)
        class_map[label_list[m] == i] = 1
        pseudo_label_cubic[:,i,:,:] = class_map

        # Build certainty cubic
        pseudo_label_cubic_certainty[:,i,:,:] = certainty_list[m][:,:,:,i]#*class_map

        # Build the mask of the cubic
        class_map = torch.zeros(img_size).cuda(gpu_id)
        for l in range(len(label_list)):
            class_map[label_list[l] == i] = 1
        pseudo_label_cubic_mask[:,i,:,:] = class_map
    
    avgPooling = nn.AvgPool2d(RF, stride=(1, 1), padding=(2, 2)).cuda(gpu_id)
    
    pseudo_label_cubic = avgPooling(pseudo_label_cubic)
    pseudo_label_cubic = pseudo_label_cubic*pseudo_label_cubic_mask
    pseudo_label_cubic_certainty = avgPooling(pseudo_label_cubic_certainty)
    pseudo_label_cubic_certainty = pseudo_label_cubic_certainty*pseudo_label_cubic_mask

    _, pseudo_label_cubic_argmax = torch.max(pseudo_label_cubic, dim=1)
    pseudo_label_cubic_argmax = pseudo_label_cubic_argmax.type(torch.ByteTensor).cuda(gpu_id)
    pseudo_label_cubic_certainty, _ = torch.max(pseudo_label_cubic_certainty, dim=1)
    pseudo_label_cubic_certainty = pseudo_label_cubic_certainty.type(torch.ByteTensor).cuda(gpu_id)
    #print(torch.max(pseudo_label_cubic_certainty))
    
    mask_ori[count_map == 1] = 1
    pseudo_label = pseudo_label*mask_ori + pseudo_label_cubic_argmax*(1-mask_ori)
    
    no_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    mask_certainty[pseudo_label_cubic_certainty >= threshold] = 1
    pseudo_label = pseudo_label*mask_certainty + no_label*(1-mask_certainty)

    return pseudo_label

def source_distribution_aggregation(priority_list, label_list, certainty_list, gpu_id, source_distribution_map=None):
    num_classes = len(priority_list)
    img_size = label_list[0].shape
    #print(img_size)
    # Generate the pseudo labels
    pseudo_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    count_map = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    # Extract the tensor
    for p in range(num_classes-1, -1, -1): # c-1 to 0
        i, m = priority_list[p][0], priority_list[p][1]
        pseudo_label[label_list[m] == i] = i
        count_map[label_list[m] == i] += 1
    
    count_mask = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    count_mask[count_map == 1] = 1
    distribution_map = torch.zeros(img_size).cuda(gpu_id)
    distribution_map = distribution_map.long()
    #print(source_distribution_map.shape, distribution_map.shape)
    for b in range(img_size[0]):
        distribution_map[b] += source_distribution_map
    distribution_map = distribution_map.type(torch.ByteTensor).cuda(gpu_id)
    pseudo_label = pseudo_label*count_mask + distribution_map*(1-count_mask)
    return pseudo_label