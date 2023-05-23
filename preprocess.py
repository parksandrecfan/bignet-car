import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import torch
import svgpathtools
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, svg2paths, svg2paths2,disvg,wsvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from wand.image import Image as WImage
import skimage.io
import skimage.filters
from google.cloud import vision
import os, io
from PIL import Image, ImageDraw, ImageFont, ImageFilter 
import matplotlib.pyplot as plt
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'silver-lead-377014-6ac0001c6c6f.json'
client=vision.ImageAnnotatorClient()
from util import dump_item, load_item, plotting

def svg2nst_path(svg_name, image_name, invert=1):#to neural style transfer path
    #use svg to png
    nst_path="jpg/%s_nst.jpg"%image_name#neural style transfer image
    nst_image=svg2png(plot=0, path=svg_name)
    if invert==1:
        Image.fromarray(np.uint8(255-np.array(nst_image)[:,:,:3])).save(nst_path)
    else:
        Image.fromarray(np.uint8(np.array(nst_image)[:,:,:3])).save(nst_path)
    return nst_path

def jpg2cst_path(image_name, new_size=(128,128), color=(0,0,0), frame=1, segment=0, background="black", threshold=3, reduced_threshold=3):#jpg to curve style transfer path
    image=image_name
    pixel_path="jpg/%s.jpg"%image
    bmp_path="bmp/%s.bmp"%image
    nx9_path="nx9/%s.pkl"%image

    potrace_svg_path="svg/%s_potrace.svg"%image
    flipped_svg_path="svg/%s_flipped.svg"%image
    cst_path="svg/%s_cst.svg"%image

    im_read=jpgpath2np(pixel_path)#pixel image to array
    if segment==1:
        im_read=segmentation(im_read, background="black", device=0)
    resize_im=np_resize(im_read,new_size=new_size,border=5, color=color) #resize an image. input np, output np
    edges_im=canny(resize_im, gx=70, gy=70)
    np2svg(edges_im, potrace_bmp=bmp_path, potrace_svg=potrace_svg_path, flipped_svg=flipped_svg_path)
    _=svg_parsing(in_path=flipped_svg_path, out_path=cst_path,\
                  option=2, print_group=0, threshold=threshold, reduced_threshold=reduced_threshold, frame=frame, color=0)
    return cst_path

#import image from path
def jpgpath2np(pixel_path="jpg/avril.jpg"):
    im_read=cv2.imread(pixel_path, cv2.IMREAD_UNCHANGED)
    im_read = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
    return im_read

#import image from path
def bmppath2np(pixel_path="jpg/avril.jpg"):
    im_read=cv2.imread(pixel_path, cv2.IMREAD_UNCHANGED)
    return im_read

def dilation_gray(uni_mask,mask_buffer):
    window_size=mask_buffer*2+1
    kernel=np.ones((window_size,window_size))
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))
    dilated_mask = torch.clamp(torch.nn.functional.conv2d(torch.Tensor(np.expand_dims(np.expand_dims(uni_mask, 0), 0)), kernel_tensor, padding=(mask_buffer,mask_buffer)), 0, 1)
    return dilated_mask[0][0]

def mask_gray2rgb(gray_mask):
    mask3d=np.zeros((gray_mask.shape[0], gray_mask.shape[1],3))
    # # np.broadcast_to(np.array(crop_mask), (crop_mask.shape[0], crop_mask.shape[1],3)).shape
    # # np.hstack((np.array(crop_mask),np.array(crop_mask))).shape
    mask3d[:,:,0]=np.array(gray_mask)
    mask3d[:,:,1]=np.array(gray_mask)
    mask3d[:,:,2]=np.array(gray_mask)
    return mask3d

def blur_mask(im_read, uni_mask, sigmas, blur_bound):
    for idx,sigma in enumerate(sigmas):
    #     print(idx,sigma)
        # apply Gaussian blur, creating a new image
        blurred = skimage.filters.gaussian(
            im_read, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

        if idx==0:
            mask2d=dilation_gray(uni_mask,blur_bound[idx])
            mask3d=np.clip(mask_gray2rgb(mask2d),0,1)
    #         mask3d=np.clip(mask_gray2rgb(dilated_list[idx]),0,1)
            img_masked=np.clip(np.multiply(blurred,mask3d)-(1-mask3d),0,1)
            img_add=np.clip(img_masked,0,1)
    #         mask_add=mask3d
        else:
            if idx==1:
                old_mask2d=mask2d
                new_mask2d=dilation_gray(uni_mask,blur_bound[idx])
            else:
                old_mask2d=new_mask2d#last mask, 就不用再算一次
                if blur_bound[idx]!=-1:
                    new_mask2d=dilation_gray(old_mask2d,blur_bound[idx]-blur_bound[idx-1])
                else:
                    new_mask2d=torch.ones(uni_mask.shape)

            mask3d=np.clip(mask_gray2rgb(new_mask2d-old_mask2d),0,1)
    #         mask3d=np.clip(mask_gray2rgb(dilated_list[idx]-dilated_list[idx-1]),0,1)
            img_masked=np.clip(np.multiply(blurred,mask3d)-(1-mask3d),0,1)
            img_add=np.clip(img_add+img_masked,0,1)
    #         mask_add=np.uint8(np.clip(mask_add+mask3d,0,255))
#         plt.figure(figsize=(10,10))  
#         plt.axis("off")
    #     print(mask3d)
    #     plt.imshow(mask3d)
#         plt.imshow(np.uint8(np.clip(img_add*255,0,255)))
    #     plt.imshow(img_masked)
#         plt.show()

    blurred_img=np.uint8(np.clip(img_add*255,0,255))
    return blurred_img

def logo_detect(in_path, frame_option=0, block_option=0, print_option=0):
    with io.open(in_path, 'rb') as image_file:
        content=image_file.read()
        
    image=vision.Image(content=content)
    response=client.logo_detection(image=image)
    logos=response.logo_annotations
    try:
        for logo in logos:
            if print_option==1:
                print('Logo decription', logo.description)
                print('Confidence', logo.score)
                
        vertices = logo.bounding_poly.vertices
        if frame_option!=0:
            frame_img = drawVertices(content, vertices, logo.description)
        else:
            frame_img = None
            
        if block_option!=0:    
            block_img = block_logo(content, vertices, logo.description)
        else:
            block_img = None
        
        return frame_img, block_img
    except:
        print("unable to detect", in_path)
        pillow_img = Image.open(io.BytesIO(content))
        return pillow_img, pillow_img
        
def drawVertices(image_source, vertices, display_text=''):
    pillow_img = Image.open(io.BytesIO(image_source))
    try:
        draw = ImageDraw.Draw(pillow_img)
        for i in range(len(vertices)-1):
            draw.line(((vertices[i].x, vertices[i].y), (vertices[i + 1].x, vertices[i + 1].y)),
                    fill='green',
                    width=8
            )

        draw.line(((vertices[len(vertices) - 1].x, vertices[len(vertices) - 1].y),
                   (vertices[0].x, vertices[0].y)),
                   fill='green',
                   width=8
        )

        font = ImageFont.truetype('font/arial.ttf', 16)
        draw.text((vertices[0].x + 10, vertices[0].y),
                  font=font, text=display_text, 
                  fill=(255, 255, 255))
    except:
        print("unable to draw")
    return pillow_img

def block_logo(image_source, vertices, display_text=''):
    try:
        pillow_img = Image.open(io.BytesIO(image_source))

        draw = ImageDraw.Draw(pillow_img)

        shape = [(vertices[0].x, vertices[0].y),(vertices[2].x,vertices[2].y)]
        draw.rectangle(shape, fill='black')
    except:
        print("unable to draw")
    return pillow_img

#simple version for this project
def segmentation_model_input(im, predictor, background="black", mode="general", buffer=5):

    outputs = predictor(im)
    outputs={key:outputs[key].to("cpu") for key in outputs}
    
    if len(outputs["instances"].pred_boxes)==0:
        print("no object detected")
        return torch.ones(im.shape[:2])


    areas=[]
    for box in outputs["instances"].pred_boxes:
        xmin, ymin, xmax, ymax=box
        area=(xmax-xmin)*(ymax-ymin)
        areas.append(area.item())
    chosen_box=areas.index(max(areas))

    mask=outputs["instances"].pred_masks[chosen_box]
    uni_mask=mask
    return uni_mask


#single mode: detect the biggest object
#general mode: detect all the objects
def segmentation(im, background="black", mode="general", buffer=5, device=0):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    outputs={key:outputs[key].to("cpu") for key in outputs}
    
    if len(outputs["instances"].pred_boxes)==0:
        print("no object detected")
        return im, torch.ones(im.shape[:2])

    if mode=="single":
        areas=[]
        for box in outputs["instances"].pred_boxes:
            xmin, ymin, xmax, ymax=box
            area=(xmax-xmin)*(ymax-ymin)
            areas.append(area.item())
        chosen_box=areas.index(max(areas))

        mask=outputs["instances"].pred_masks[chosen_box]
        uni_mask=mask
    
    if mode=="general":
        mask=outputs["instances"].pred_masks
        uni_mask=torch.sign(torch.sum(mask, dim=0))
        # plt.imshow(uni_mask)
    
    #layers of dilation
#     dilated_list=[]
#     dilated_list.append(dilation_gray(uni_mask,mask_buffer+2))
#     dilated_list.append(dilation_gray(uni_mask,mask_buffer+4))
#     dilated_list.append(dilation_gray(uni_mask,mask_buffer+6))
#     dilated_list.append(dilation_gray(uni_mask,mask_buffer+8))
#     dilated_list.append(torch.ones(uni_mask.shape))
    
#     #dilation on uni_mask
#     dilated_mask=dilation_gray(uni_mask,mask_buffer)

#     uni_mask=dilated_mask
    
    xmin=max(torch.where(uni_mask==1)[0].min().item()-buffer, 0)
    # print("xmin",xmin)
    xmax=min(torch.where(uni_mask==1)[0].max().item()+buffer, im.shape[0])
    # print("xmax",xmax)
    ymin=max(torch.where(uni_mask==1)[1].min().item()-buffer, 0)
    # print("ymin",ymin)
    ymax=min(torch.where(uni_mask==1)[1].max().item()+buffer, im.shape[1])
    # print("ymax",ymax)
    # cropped_mask=uni_mask[xmin:xmax,ymin:ymax] #I crop it with image, so this is not used

#     def mask_gray2rgb(gray_mask):
#         mask3d=np.zeros((gray_mask.shape[0], gray_mask.shape[1],3))
#         # # np.broadcast_to(np.array(crop_mask), (crop_mask.shape[0], crop_mask.shape[1],3)).shape
#         # # np.hstack((np.array(crop_mask),np.array(crop_mask))).shape
#         mask3d[:,:,0]=np.array(gray_mask)
#         mask3d[:,:,1]=np.array(gray_mask)
#         mask3d[:,:,2]=np.array(gray_mask)
#         return mask3d
    
    mask3d=mask_gray2rgb(uni_mask)
#     mask3d=np.zeros((uni_mask.shape[0], uni_mask.shape[1],3))
#     # # np.broadcast_to(np.array(crop_mask), (crop_mask.shape[0], crop_mask.shape[1],3)).shape
#     # # np.hstack((np.array(crop_mask),np.array(crop_mask))).shape
#     mask3d[:,:,0]=np.array(uni_mask)
#     mask3d[:,:,1]=np.array(uni_mask)
#     mask3d[:,:,2]=np.array(uni_mask)

    if background==None:
        return uni_mask
    
    elif background=="white":
        #white background
        mask_im=np.uint8(np.clip(np.multiply(im, mask3d)+255*(1-mask3d),0,255))
    elif background=="black":
        #black background
        mask_im=np.uint8(np.clip(np.multiply(im, mask3d)+255*-(1-mask3d),0,255))
    elif background=="blur":
        #parameters for blurry
        sigmas = [0,2,5,10,12,14,16,18,20,22,24,26]
#         mask_buffer=0
        blur_bound=[0,1,3,5,10,15,20,25,30,35,40,-1]
        mask_im=blur_mask(im, uni_mask, sigmas, blur_bound)
    # plt.imshow(mask_im)

    #crop image with bounding box information
    crop_im=mask_im[xmin:xmax,ymin:ymax]
    return crop_im, uni_mask

#2. resize

def np_resize(im,new_size=(128,128),border=5, mode="constant", color=0):#input numpy, output numpy
    goal_h, goal_w=new_size
    goal_w=goal_w-border*2
    goal_h=goal_h-border*2
    h,w,_=im.shape
    if h>w:
        scaled_size=(int(goal_h/h*w),goal_h)
    else:
        scaled_size=(goal_w, int(goal_w/w*h))
    resize_im=Image.fromarray(im).resize(scaled_size)
#     plt.axis("off")
    
    

#     def add_margin(pil_img, top, right, bottom, left, color):
#         width, height = pil_img.size
#         new_width = width + right + left
#         new_height = height + top + bottom
#         result = Image.new(pil_img.mode, (new_width, new_height), color)
#         result.paste(pil_img, (left, top))
#         return result

    # print(resize_im.size)
    resized_w, resized_h=resize_im.size
    top_pad=int((goal_h-resized_h)/2)+border
    bottom_pad=goal_h-resized_h-top_pad+border*2
    left_pad=int((goal_w-resized_w)/2)+border
    right_pad=goal_w-resized_w-left_pad+border*2
    
    resize_im=np.array(resize_im)
    pad=((top_pad, bottom_pad), (left_pad, right_pad),(0,0))
    if mode=="constant":
        im_padded=np.pad(resize_im, pad, mode=mode, constant_values=color)
    if mode=="edge":
#         print(resize_im.shape)
#         print(pad)
        im_padded=np.pad(resize_im, pad, mode=mode)
    # color=(128,0,64)
    # color=(0,0,0)
#     im_padded = add_margin(resize_im, top_pad, right_pad, bottom_pad, left_pad, color)
    # plt.imshow(im_new)
    #resize the bigger height/width, pad the rest
#     resize_im=np.array(im_padded)
    
    bbox=[left_pad, bottom_pad, left_pad+resized_w, bottom_pad+resized_h]
    return im_padded, bbox

#3. canny edge
def canny(im, gx=70,gy=70,mask=0, bbox=None):
    edges_im = cv2.Canny(im,gx,gy,False)
    if mask==1:
        x0,y0,x1,y1=bbox
        mask=np.zeros(edges_im.shape)
        mask[y0:y1,x0:x1]=1
        edges_im=np.multiply(mask,edges_im)
    return edges_im

def svg_pic2pathsum(path, vflip=0):
    eps=1e-10
    paths, attributes,svg_attributes=svg2paths2(path)
    svgpath=0
    svgpathsum=0
    for i in paths:
        for j in i:
            if isinstance(j, Arc):
            
                xc=j.center.real
                yc=j.center.imag
                x1=j.start.real
                y1=j.start.imag
                x4=j.end.real
                y4=j.end.imag

                ax = x1 - xc
                ay = y1 - yc
                bx = x4 - xc
                by = y4 - yc
                q1 = ax * ax + ay * ay
                q2 = q1 + ax * bx + ay * by
                k2 = (4/3) * ((np.sqrt(2 * q1 * q2) - q2)+eps) / ((ax * by - ay * bx)+eps)
                
                if j.sweep==False and len(i)==2:
#                     print("here2")
                    c0=complex(x4,y4)
                    c3=complex(x1,y1)
                    x2 = xc - ax + k2 * ay
                    y2 = yc - ay - k2 * ax
                    x3 = xc - bx - k2 * by                                 
                    y3 = yc - by + k2 * bx
                    c1=complex(x2,y2)
                    c2=complex(x3,y3)
                else:
                    x2 = xc + ax - k2 * ay
                    y2 = yc + ay + k2 * ax
                    x3 = xc + bx + k2 * by                                 
                    y3 = yc + by - k2 * bx
                    c0=complex(x1,y1)
                    c1=complex(x2,y2)
                    c2=complex(x3,y3)
                    c3=complex(x4,y4)
 
            if isinstance(j, Line):
                c0, c1, c2, c3=j.start, j.start*2/3+j.end/3, j.start/3+j.end*2/3, j.end
            if isinstance(j, QuadraticBezier):
                c0, c1, c2, c3=j.start, j.start/3+j.control*2/3, j.control*2/3+j.end/3, j.end
            if isinstance(j, CubicBezier):
                c0, c1, c2, c3=j.start, j.control1, j.control2, j.end
                
            if vflip==1:
                c0=complex(c0.real,-c0.imag)
                c1=complex(c1.real,-c1.imag)
                c2=complex(c2.real,-c2.imag)
                c3=complex(c3.real,-c3.imag)
                
            seg=CubicBezier(c0,c1,c2,c3)
            
            if svgpath==0:
                svgpath=Path(seg)
            else:
                svgpath.append(seg)
        if svgpathsum==0:
            svgpathsum=svgpath
            oldsvgpath=svgpath
        else:
            if svgpathsum==oldsvgpath:
                svgpathsum=[svgpathsum,svgpath]
            else:
                svgpathsum.extend(Path(svgpath))
        svgpath=0
    return svgpathsum

# potrace

def np2svg(edges_im,potrace_bmp="avril.bmp", potrace_svg="temp.svg", flipped_svg="avril_good.svg"):
    #first, turn into 1D image->bmp
    cv=Image.fromarray(np.uint8(edges_im), 'L')
    cv.save("%s"%potrace_bmp)
#     get_ipython().system('~/potrace_exe/potrace "%s" -t 10 -a 1 -o "%s" -b svg'%(potrace_bmp, potrace_svg))
    get_ipython().system('~/potrace_exe/potrace "%s" -t 10 -a 1 -o "%s" -b svg'%(potrace_bmp, potrace_svg))
    #because some svgpathtool defect, need to vflip once before proceeding

    svg_vflip=svg_pic2pathsum("%s"%potrace_svg, vflip=1)
    wsvg(svg_vflip,filename="%s"%flipped_svg)#good means vertically resolved
    
def svg_parse(gray_svg_path="avril_good.svg", color_svg_path="avril_color.svg", option=2, threshold=10, reduced_threshold=3, frame=0):
    _=svg_parsing("%s"%gray_svg_path, "%s"%color_svg_path, option=option, print_group=1, \
                      threshold=threshold, reduced_threshold=reduced_threshold, frame=frame)
    
#input: path, output: colors
def path2color(path):
    color_sum=[]
    for i in range(len(path)):
        now1=np.random.randint(0,255)
        now2=np.random.randint(0,255)
        now3=np.random.randint(0,255)
        colors=now1,now2,now3
        color_sum.append(colors)
        
    return color_sum

    
def svg_parsing(in_path, out_path, option=2, print_group=1, threshold=0, reduced_threshold=0, frame=1, color=1):
   
    paths, attributes = svg2paths(in_path)
    if option!=0:
        paths_self_group=[]
        paths_self_sum=[]
        for i in paths:
            first_in_group=1
            if len(i)>threshold:
                
                for jdx, j in enumerate(i):
                    #if first, auto append
                    if first_in_group==1: #if this is the first curve in a group
                        paths_self_group=Path(j)#start a new path
                        #if the first curve is the last curve:
                        if len(i)==1:#if there is only one curve
                            paths_self_sum.append(paths_self_group)#add this group to sum
                            first_in_group=1
                        else:
                            first_in_group=0
                    elif jdx!=len(i)-1:#if not the last curve
                        if j.end==i[jdx+1].start:
                            paths_self_group.append(j)
                            first_in_group=0
                        else:
                            paths_self_group.append(j)
                            paths_self_sum.append(paths_self_group)
                            first_in_group=1
                    else:
                        paths_self_group.append(j)
                        paths_self_sum.append(paths_self_group)
                        first_in_group=1
                        
        paths=paths_self_sum
        
        if option==2:
            reduced_group=[]
            for i in paths_self_sum:
                if len(i)>reduced_threshold:
                    reduced_group.append(i)

            paths=reduced_group
    
    if print_group==1:
        print("number of groups: ", len(paths))
        tot_crvs=0
        for i in paths:
            tot_crvs+=len(i)
        print("number of curves(total): %s"%tot_crvs)
    
    #this is a cheaty way to get rid of the frame...
    if frame==0:#first path is always frame
        if color==1:
            color_sum=path2color(paths[1:])
            wsvg(paths[1:], colors=color_sum, filename=out_path)
        else:
            wsvg(paths[1:], filename=out_path)
        return paths[1:]

    #this is what I am supposed to do
    if color==1:
        color_sum=path2color(paths)
        wsvg(paths, colors=color_sum, filename=out_path)
    else:
        wsvg(paths, filename=out_path)
    return paths



def svg2nx9(svg_path, nx9_path, print_option=0, vflip=0):
    svg_orig=svg_pic2pathsum(svg_path, vflip=vflip)#already flipped, no need to do it again
    nx9=[]
    curve_list=[]

    for idx,i in enumerate(svg_orig):#a idx is a group
        for jdx,j in enumerate(i):#a jdx is a curve
            for kdx,k in enumerate(j):#a k is a point
                curve_list.append(k.real)
                curve_list.append(k.imag)
            curve_list.append(idx)
            nx9.append(curve_list)
            curve_list=[]

    #4. save and load the nx9 structure
    item2dump=np.array(nx9)
    dump_path=nx9_path
    dump_item(item2dump, dump_path)
    nx9_loaded=load_item(dump_path)
    
    if print_option==1:
        print("nx9 shape:",nx9_loaded.shape)
    return nx9_loaded

#svg2png
# def svg2png(res=100,plot=1, path='test.svg'):

#     car_drawing = svg2rlg(path)
#     renderPDF.drawToFile(car_drawing,'test.pdf')
#     display_im = WImage(filename='test.pdf',resolution=res)
#     os.remove("test.pdf")
#     if plot==1:
#         plotting(display_im)
#     return display_im

#nx9 to svg
def nx92svg(pic,#one nx9 picture
                filter=0,
                print_option=1,
                color=1,
                res=100,
                scale=1,
                save_at="test.svg"):

    pic_group=np.array(pic[:,-1])
    i=min(pic[:,-1])#1
    svgpath=0
    svgpathsum=0
    cancel_factor=filter
    while i<max(pic[:,-1])+1:#12
        min_loc=np.amin(np.where(pic_group==i))
        max_loc=np.amax(np.where(pic_group==i))
        c0_x=pic[min_loc][0]
        c0_y=pic[min_loc][1]
#         c0_x=round(c0_x,2)
#         c0_y=round(c0_y,2)
        if max_loc-min_loc>cancel_factor:
            j=min_loc
            while j<max_loc+1:
                end_point_x, end_point_y = pic[j][6],pic[j][7]

                c1_x, c1_y = pic[j][2],pic[j][3]
                c2_x, c2_y = pic[j][4],pic[j][5]
                try:
#                     c1_x=round(c1_x,2)
#                     c1_y=round(c1_y,2)
#                     c2_x=round(c2_x,2)
#                     c2_y=round(c2_y,2)
#                     end_point_x=round(end_point_x,2)
#                     end_point_y=round(end_point_y,2)
                    c0=complex(c0_x,c0_y)
                    c1=complex(c1_x,c1_y)
                    c2=complex(c2_x,c2_y)
                    c3=complex(end_point_x,end_point_y)
                    seg= CubicBezier(c0,c1,c2,c3)
                    if svgpath==0:
                        svgpath=Path(seg)
                    else:
                        svgpath.append(seg)
                except:
                    pass
                c0_x=end_point_x
                c0_y=end_point_y
#                 c0_x=round(c0_x,2)
#                 c0_y=round(c0_y,2)
                j+=1
            if svgpathsum==0:
                svgpathsum=svgpath
                oldsvgpath=svgpath
            else:
                if svgpathsum==oldsvgpath:
                    svgpathsum=[svgpathsum,svgpath]
                else:
                    svgpathsum.extend(Path(svgpath))
            svgpath=0
        i+=1
    sum=len(pic)

    if print_option==1:
        from IPython.display import SVG
        from cairosvg import svg2png
        if color==1: 
            colortagsum=path2color(svgpathsum)
            wsvg(svgpathsum,colors=colortagsum,filename=save_at)
            
            a=SVG(save_at)
            svg2png(bytestring=a.data,write_to='output.jpg',dpi=res, scale=scale)
            im = Image.open("output.jpg") 
            plotting(im, figsize=(15,15))
#             svg2png(res=res, path=save_at, plot=1)
#             SVG(save_at)
        else:
            wsvg(svgpathsum,filename=save_at)  
            
            a=SVG(save_at)
            svg2png(bytestring=a.data,write_to='output.jpg',dpi=res, scale=scale)
            im = Image.open("output.jpg") 
            plotting(im, figsize=(15,15))
#             svg2png(res=res, path=save_at, plot=1)
#             SVG(save_at)

#     if print_option==1:
# #         from IPython.display import SVG
#         if color==1: 
#             colortagsum=path2color(svgpathsum)
#             wsvg(svgpathsum,colors=colortagsum,filename=save_at)
#             svg2png(res=res, path=save_at, plot=1)
# #             SVG(save_at)
#         else:
#             wsvg(svgpathsum,filename=save_at)  
#             svg2png(res=res, path=save_at, plot=1)
# #             SVG(save_at)
    else:
        if color==1: 
            colortagsum=path2color(svgpathsum)
            wsvg(svgpathsum,colors=colortagsum,filename=save_at)
        else:
            wsvg(svgpathsum,filename=save_at)  

    k=0
    sum=0
    while k<len(svgpathsum):
        sum+=len(svgpathsum[k])
        k+=1
    
    if print_option==1:
        if sum/len(svgpathsum)==4:
            print("groups=1 bezier curves:",len(svgpathsum))
        else:
            print("groups=",len(svgpathsum),"bezier curves:",sum)
    return svgpathsum

def svg2gnn(picpath, print_option=1, h_aug=0, v_aug=0, normalize=1):
    svg_orig=svg_pic2pathsum(path=picpath)
    #check if there is only one group, if it is and []
    if str(type(svg_orig[0]))=="<class 'svgpathtools.path.CubicBezier'>":
        svg_orig=[svg_orig]
        
    #this part is to find all the bounding boxes
    xmin=0
    xmax=0
    ymin=0
    ymax=0
    bbox=[]
    for idx,i in enumerate(svg_orig):#and idx is a group
    #         print(idx+1,"group has",len(i),"curves")
        for jdx,j in enumerate(i):#a jdx is a curve
            for kdx,k in enumerate(j):#a kdx is an x or y component
                if(jdx==0 and kdx==0):#if it is the first curve in a group
                    xmin=k.real
                    xmax=k.real
                    ymin=k.imag
                    ymax=k.imag
                else:
                    xmin=k.real if float(k.real)<xmin else xmin
                    xmax=k.real if float(k.real)>xmax else xmax
                    ymin=k.imag if float(k.imag)<ymin else ymin
                    ymax=k.imag if float(k.imag)>ymax else ymax
        bbox.append([xmin,ymin,xmax,ymax])
    #         curves+=len(i)
    
    #find the biggest bounding box
    xmin=0
    ymin=0
    xmax=0
    ymax=0
    for idx,i in enumerate(bbox):
        xmin=i[0] if idx==0 else i[0] if i[0]<xmin else xmin
        ymin=i[1] if idx==0 else i[1] if i[1]<ymin else ymin
        xmax=i[2] if idx==0 else i[2] if i[2]>xmax else xmax
        ymax=i[3] if idx==0 else i[3] if i[3]>ymax else ymax

    #     print(xmin,ymin,xmax,ymax)
    #find the height and center xy
    h=ymax-ymin#need to keep this! add back at final representation
    absh=h
    cx=(xmin+xmax)/2
    cy=(ymin+ymax)/2
    
    #normalize h to 1 and move center to 0,0.
    #do it on bbox separately to save computational cost
    for idx,i in enumerate(bbox):
        for jdx,j in enumerate(i):
            if normalize==1:
                i[jdx]=(i[jdx]-cx)/h if jdx%2==0 else i[jdx] #odd,  x
                i[jdx]=(i[jdx]-cy)/h if jdx%2==1 else i[jdx] #even, y
            elif normalize==0:
                pass
    #     bbox

    #2. shift to origin and make svg to list 1*8
    svg_list=[]
    group_list=[]
    curve_list=[]

    #augmentation
    h_shift=0
    v_shift=0
    if h_aug==1:
        h_shift=random.uniform(-0.05,0.05)
    if v_aug==1:
        v_shift=random.uniform(-0.05,0.05)    

    # random.uniform(0,1)
    for idx,i in enumerate(svg_orig):#a idx is a group
        for jdx,j in enumerate(i):#a jdx is a curve
            for kdx,k in enumerate(j):#a k is a point
                if normalize==1:
                    curve_list.append((k.real-cx)/h+h_shift)#normalize
                    curve_list.append((k.imag-cy)/h+v_shift)#normalize
                elif normalize==0:
                    curve_list.append(k.real)
                    curve_list.append(k.imag)
            group_list.append(curve_list)
            curve_list=[]
        svg_list.append(group_list)
        group_list=[]

    bbox_par=[]
    
    #calculate bounding box parameters for correlation matrix
    for idx,i in enumerate(bbox):#every box(xmin,ymin,xmax,ymax)
        [xmin,ymin,xmax,ymax]=i
        area=(xmax-xmin)*(ymax-ymin)
        cx=(xmin+xmax)/2
        cy=(ymin+ymax)/2
        w=xmax-xmin
        h=ymax-ymin
        bbox_par.append([area,cx,cy,w,h])

    #sort with the area information
    bbox_par, svg_list=zip(*sorted(zip(bbox_par, svg_list), key=lambda x: x[0][0])[::-1])
    #print the first bounding box: is it the frame?
    if print_option==1:
        print("[a,   cx,   cy,   w,   h]")
        for i in bbox_par:
            for j in i:
                print(round(j,2),end=', ')
            print()    
        print()
        print()

    cor=np.zeros((len(bbox_par),len(bbox_par),len(bbox_par[0])))
    for n,corn in enumerate(cor):
        for m,corm in enumerate(corn):
            # new version: intuitively, large ratio should give large value
            cor[m][n][0]=bbox_par[n][0]/bbox_par[m][0]
            cor[m][n][1]=bbox_par[n][1]-bbox_par[m][1]
            cor[m][n][2]=bbox_par[n][2]-bbox_par[m][2]
            cor[m][n][3]=bbox_par[n][3]/bbox_par[m][3]
            cor[m][n][4]=bbox_par[n][4]/bbox_par[m][4]
            #old version
            # cor[n][m][0]=bbox_par[n][0]/bbox_par[m][0]
            # cor[n][m][1]=bbox_par[n][1]-bbox_par[m][1]
            # cor[n][m][2]=bbox_par[n][2]-bbox_par[m][2]
            # cor[n][m][3]=bbox_par[n][3]/bbox_par[m][3]
            # cor[n][m][4]=bbox_par[n][4]/bbox_par[m][4]
            
    if normalize==1:
        # new version: ratios take log, so itself will be 0,0,0,0,0
        cor[:,:,0]=np.log(cor[:,:,0])
        cor[:,:,3]=np.log(cor[:,:,3])
        cor[:,:,4]=np.log(cor[:,:,4])
        # old version, doesnt make much sense
        # cor=cor/cor.sum(axis=0) #make the sum 1, not sure if is a good idea
    return svg_list, cor

def nx92gnn(nx9_path, print_option=1, svg_path="test.svg", normalize=1, rules=None):
    nx9=load_item(nx9_path)
    if rules!=None:
        pass
    svgpathsum=nx92svg(nx9, filter=0, print_option=0, color=0, res=100, scale=1, save_at=svg_path)
    wsvg(svgpathsum, filename=svg_path)
    svg_list, cor=svg2gnn(svg_path, print_option=print_option, h_aug=0, v_aug=0, normalize=normalize)
    return svg_list, cor


def curve_labeling(this_item, temp_seed=0):
    #turn a list of svg pictures into flat tensors

    curve_label=[] 
    item_flat=[]
    curve_base, group_base=0,0
    for idx,i in enumerate(this_item):#pic_num
        item_idx=1
        for jdx,j in enumerate(i):#group_num
            item_jdx=1
            for kdx,k in enumerate(j):#curve_num
                curve_label.append([item_idx, item_jdx])
                item_idx=0
                item_jdx=0
                item_flat.append(k)
    return torch.FloatTensor(item_flat), np.array(curve_label) # a tensor and a np

def paths2brand(paths):
    labels=[]
    for name in paths:
        brand=int(name.split("/")[-1].split("-")[0])
        if brand==39:#toyota
            labels.append(0)
        elif brand==73:#volkswagen
            labels.append(1)
        elif brand==77:#Benz
            labels.append(2)
        elif brand==78:#Audi
            labels.append(3)
        elif brand==81:#BMW
            labels.append(4)
        elif brand==118:#Hyundai
            labels.append(5)
        elif brand==97:#Nissan
            labels.append(6)
        elif brand==122:#Ford
            labels.append(7)
        elif brand==140:#KIA
            labels.append(8)
        elif brand==157:#chevy
            labels.append(9)
    return labels