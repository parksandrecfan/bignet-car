import pickle
import matplotlib.pyplot as plt
from PIL import Image
import pynvml
import torch
import numpy as np
import glob
import os
from IPython.display import HTML
import base64
import shutil

def dump_item(this_item, path):
    open_file = open(path, "wb")
    pickle.dump(this_item, open_file)
    open_file.close()
    
def load_item(path):
    open_file = open(path, "rb")
    this_item=pickle.load(open_file)
    open_file.close()
    return this_item

def plotting(pic, figsize=(10,10), cmap=None):
    plt.figure(figsize=figsize)  
    plt.axis("off")
    if cmap!=None:
        plt.imshow(pic, cmap=cmap)
    else:
        plt.imshow(pic)
    plt.show()

def get_filelist(dir, path, include=None, exclude=None):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if include!=None and exclude==None:
                if include in filename:
                    Filelist.append(os.path.join(home, filename))
                    
            elif include==None and exclude==None:
                Filelist.append(os.path.join(home, filename))
                
            elif  include!=None and exclude!=None:
                if include in filename and exclude not in filename:
                    Filelist.append(os.path.join(home, filename))
            else:
                if exclude not in filename:
                    Filelist.append(os.path.join(home, filename))
    return Filelist
    
# import grammars as g
import preprocess as p

def animate(start_nx9, rule_name, rule, params, res=30, duration=50, gif_path=None, plot=1):
    gif_folder='gif/%s'%rule_name
    shutil.rmtree(gif_folder)
    os.mkdir(gif_folder)
    rules_applied=0
    for param in params:
        modified_nx9=rule(start_nx9, param)
        rules_applied+=1
        svg_at='temp.svg'
        _=p.nx92svg(pic=modified_nx9, print_option=0, color=0, save_at=svg_at)
        img=p.svg2png(res=res, plot=0, path=svg_at)
        img.save(filename='%s/%s.jpg'%(gif_folder,rules_applied))

    folder2gif('%s/*.jpg'%gif_folder, gif_path, duration=duration, plot=plot)
    os.remove(svg_at)
    
def folder2gif(fp_in,fp_out, duration=100, plot=1):
    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime))#sort by modified time
    # https://stackoverflow.com/questions/6773584/how-are-glob-globs-return-values-ordered
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)
    
    if plot==1:
        b64 = base64.b64encode(open(fp_out,'rb').read()).decode('ascii')
        display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))
    
def plot_process(start_canvas, goal_canvas, figsize=(6,8), label=["learn","goal"], cmap="gray"):
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.imshow(np.sign(start_canvas.detach().numpy()), cmap="gray")
    plt.title(label[0])
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.imshow(np.sign(goal_canvas.detach().numpy()), cmap="gray")
    plt.title(label[1])
    
def sparse_pixel_loss(learn_pic, goal_pic, thres=0.9):
    learn=torch.nn.Softsign()(learn_pic)
    goal=torch.nn.Softsign()(goal_pic)
    
    edge_num=torch.sum(goal>thres)#number of pixels that have something
    edge_correctness=torch.sum(learn*(goal>thres))/edge_num#pixels that are learned correctly
    return -edge_correctness*100+nn.MSELoss()(learn, goal)


def pixel_loss(learn_pic, goal_pic):
    learn=torch.nn.Softsign()(learn_pic)
    goal=torch.nn.Softsign()(goal_pic)
    loss=torch.sum(abs(learn-goal))/learn_pic.shape[0]/learn_pic.shape[1]
    # loss=torch.sum(abs(learn-goal))#for speed
    return loss    

def sub(img):
    return img[0].permute(1, 2, 0).detach().cpu().numpy()

#plot pixel base result
def plot2(left_jpg, right_jpg, figsize=(6,8), label=["learn","goal"], cmap="gray"):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.axis("off")
    plt.title(label[0])
    plt.imshow(left_jpg, cmap=cmap)
    plt.subplot(122)
    plt.axis("off")
    plt.title(label[1])
    plt.imshow(right_jpg, cmap=cmap)
    plt.show()

#check gpu

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free / 1024 **3

def cuda_print(device=0, string='0'):
    # print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')
    print(string+'reserved:   ', round(torch.cuda.memory_reserved(device)/1024**3,1), 'GB')
    # total=round(get_memory_free_MiB(device),1)
    # print('left:   ', round(total-torch.cuda.memory_reserved(device)/1024**3,1), 'GB')
    

