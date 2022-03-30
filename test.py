"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from genericpath import exists
import os
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import copy
import torch
import cv2
import numpy as np
import torchvision
# root='/train/xinying/inpainting/val/FFHQ/val_seg/'
# folder_list=os.listdir(root)
# for folder in folder_list:
#     img_list=os.listdir(os.path.join(root,folder))
#     for img in img_list:
#         sub=img.split('_')
#         sub=sub[0].split('.')
#         cmd = 'mv '+root+folder+'/'+img+' '+root+folder+'/'+sub[0]+'.png'
#         os.system(cmd)

# root='/train/xinying/inpainting/val/FFHQ/val_mask/'
# folder_list=os.listdir(root)
# for folder in folder_list:
#     img_list=os.listdir(os.path.join(root,folder))
#     for img in img_list:
#         sub=img.split('_')
#         sub=sub[0].split('.')
#         cmd = 'mv '+root+folder+'/'+img+' '+root+folder+'/'+sub[0]+'.png'
#         os.system(cmd)


# root='/train/xinying/inpainting/face/image/CelebA-512x512'
# img_list=os.listdir(root)
# for img in img_list:
#     sub=img.split('_')
#     image_path='/train/xinying/inpainting/face/image/CelebA-512x512/'+sub[0]
#     seg_path= '/train/xinying/inpainting/face/seg/CelebA-seg/'+sub[0]       
#     image=cv2.imread(image_path)
#     seg=cv2.imread(seg_path)
#     tmp=image*0.3+seg*3
#     cv2.imwrite('test.png',tmp)


# test
def inference_crop(data):

# crop image into several 512x512 pieces and 
    image=data['image']
    label=data['label']
    mask=data['mask'] 
    
    ori_shape2=image.shape[2]
    ori_shape3=image.shape[3]


    w_p=0
    h_p=0
    if ori_shape2<512 or ori_shape3<512:  
        if ori_shape2<512:
            h_p=int(np.ceil((512-image.shape[2])/2)) 
        if ori_shape3<512:  
            w_p=int(np.ceil((512-image.shape[3])/2))
        
        trans=torchvision.transforms.Pad(padding=[w_p,h_p], padding_mode='reflect')
        image=trans(image)
        label=trans(label)
        mask=trans(mask)

    row=image.shape[2]
    col=image.shape[3]

    row_num = int(np.ceil(row/512))
    y_step = int(np.ceil(row_num*512-row)/row_num)
    if y_step==0 and row_num>1:
        row_num=row_num+1
        y_step = int(np.ceil(row_num*512-row)/row_num)
    
    col_num = int(np.ceil(col/512))
    x_step = int(np.ceil(col_num*512-col)/col_num)
    if x_step==0 and col_num>1:
        col_num=col_num+1
        x_step = int(np.ceil(col_num*512-col)/col_num)

    canvas=image*0
    canvas_count=image*0
    ymin=0
    ymax=0 
    xmin=0
    xmax=0

    while True:
        ymax=ymin+512
        if ymax>row:
            # reach row maximum
            ymax=row
            ymin=ymax-512

        while True:
            xmax=xmin+512
            if xmax>col:
                # reach col maximum
                xmax=col
                xmin=xmax-512

            image_crop=image[:,:,ymin:ymax,xmin:xmax]
            label_crop=label[:,:,ymin:ymax,xmin:xmax]
            mask_crop=mask[:,:,ymin:ymax,xmin:xmax]
            data['image']=image_crop
            data['label']=label_crop
            data['mask']=mask_crop
            generated = model(data, mode='inference') 
            generated=generated.cpu()
            
            ymin_s=0
            ymax_s=0
            xmin_s=0
            xmax_s=0
            if ymin!=0:
                ymin_s=min(10,y_step)
            if ymax!=row:
                ymax_s=min(10,y_step)
            if xmin!=0:
                xmin_s=min(10,x_step)
            if xmax!=col:
                xmax_s=min(10,x_step)
            
            # optimize edge  
            canvas[:,:,ymin+ymin_s:ymax-ymax_s,xmin+xmin_s:xmax-xmax_s]+=generated[:,:,ymin_s:512-ymax_s,xmin_s:512-xmax_s]
            canvas_count[:,:,ymin+ymin_s:ymax-ymax_s,xmin+xmin_s:xmax-xmax_s]+=1   
             
            xmin=xmin+512-x_step
            if xmax>=col:
                ymin=ymin+512-y_step
                break
        if ymax>=row:
            break

    image=canvas/canvas_count

    image=image[:,:,h_p:row-h_p,w_p:col-w_p]
                     
    image=torch.nn.functional.interpolate(image,size=[ori_shape2,ori_shape3])            
    image=(image.cpu().numpy().squeeze(0).transpose((1, 2, 0))+1)/2*255
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=(image+0.5).astype(int)

    return image

def inference_resize(data, copy_ori=False):
# resize to [512,512]
    
    image_ori=data['image']
    mask_ori=data['mask']
    mask_rgb=copy.deepcopy(image_ori)
    mask_rgb[:,0,:,:]=mask_ori
    mask_rgb[:,1,:,:]=mask_ori
    mask_rgb[:,2,:,:]=mask_ori

    image=data['image']
    label=data['label']
    mask=data['mask']

    ori_shape2=image.shape[2]
    ori_shape3=image.shape[3]
    
    new_shape2=512
    new_shape3=512
    if not (image.shape[2]==new_shape2 and image.shape[3]==new_shape3):      
        image=torch.nn.functional.interpolate(image,size=[new_shape2,new_shape3])            
        label=torch.nn.functional.interpolate(label,size=[new_shape2,new_shape3],mode='nearest')
        mask=torch.nn.functional.interpolate(mask,size=[new_shape2,new_shape3],mode='nearest')
    
    data['image']=image
    data['label']=label
    data['mask']=mask 
    
    generated = model(data, mode='inference') 
    generated=generated.cpu()

    image=torch.nn.functional.interpolate(generated,size=[ori_shape2, ori_shape3])
    if copy_ori:    
        image_ori[mask_rgb==1]=image[mask_rgb==1]  
        image=image_ori  

    image=(image.cpu().numpy().squeeze(0).transpose((1, 2, 0))+1)/2*255
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=(image+0.5).astype(int)

    return image



opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt)
model.eval()


if 'ffhq' in opt.image_dir.lower():
    if 'val' in opt.image_dir.lower():
        root='./result/FFHQ/val/'
    if 'test' in opt.image_dir.lower():
        root='./result/FFHQ/test/'
if 'places' in opt.image_dir.lower():
    if 'val' in opt.image_dir.lower():
        root='./result/Places/val/'
    if 'test' in opt.image_dir.lower():
        root='./result/Places/test/'


if not exists(root):
    os.makedirs(root)
    
for i, data_i in enumerate(dataloader):
    
    data=copy.deepcopy(data_i)
    
    ori_shape2=data['image'].shape[2]
    ori_shape3=data['image'].shape[3]
    
    type=data['path'][0].split('/')[7]
    name=data['path'][0].split('/')[8]
    
    # if not name[5]=='0':
    #     continue    

    if not exists(os.path.join(root,type)):
        os.mkdir(os.path.join(root,type))
    
    image_out_path=root+type+'/'+name

    if type=='Nearest_Neighbor' or type=='Every_N_Lines':
        result=inference_crop(data_i)
    else:
        result=inference_resize(data_i, True)

    cv2.imwrite(image_out_path, result)



