import preprocess as p

def predict1svg(model, svg_path, print_option=1):

    svg_list, cor= p.svg2gnn(picpath=svg_path, print_option=0, normalize=1)
    curve_tensor, curve_label=p.curve_labeling([svg_list])
    
    model.eval()
    pred=model(curve_tensor, curve_label, [cor])
    pred_brand=pred.detach().numpy().argmax(axis=1)[0]
    true_brand=p.paths2brand([svg_path])[0]
    
    right= (true_brand == pred_brand)
    if print_option==1:
        if right == 0:
            print("true: %s, predict: %s, (%s)"%(true_brand, pred_brand, "wrong"))
        if right == 1:
            print("true: %s, predict: %s, (%s)"%(true_brand, pred_brand, "right"))
    return right

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def vec2plot(latent_vec, mode="tsne", dimension=2):
    if mode=="tsne":
        model = TSNE(n_components=dimension, random_state=0)
    if mode=="pca":
        model = PCA(n_components=dimension)

    vector2plot=model.fit_transform(latent_vec) 
    return vector2plot

import numpy as np
import util as u

def get_nx9_group_rec(model, nx9, true_brand, temp_path="test.pkl", svg_folder="test/", print_option=1, cache=0):
    start_group=0
    end_group=nx9[-1,-1]#last group index
    group_rec=[]# 記錄原始信心數據

    for del_group in range(int(end_group)+1):
        # del_group=2
        before=nx9[np.where(nx9[:,-1]<del_group)]
        after=nx9[np.where(nx9[:,-1]>del_group)]
        after[:,-1]=after[:,-1]-1#move the groups forward

        ablation_group=np.vstack((before,after))
    #     np.vstack((before,after)).shape


        # check how acc change
        u.dump_item(ablation_group, temp_path)
        #for debug, not really useful
#         svg_list2, cor2=p.nx92gnn(temp_path, print_option=0, \
#                                   svg_path=svg_folder+"group %s.svg"%del_group)
        svg_list2, cor2=p.nx92gnn(temp_path, print_option=0, \
                                  svg_path=svg_folder+"temp%s.svg"%cache)
        curve_tensor2, curve_label2=p.curve_labeling([svg_list2])

        pred_ablation=model(curve_tensor2, curve_label2, [cor2])
    #     pred_brand_ablation=pred_ablation.detach().numpy().argmax(axis=1)[0]
        # true_brand=p.paths2brand([svg_path])[0]#done already
        confidence=pred_ablation.detach().numpy()[0][true_brand]
        group_rec.append(confidence)
        if print_option==1:
            print(del_group, confidence)
        
    return group_rec


def get_nx9_curve_rec(model, nx9, true_brand, temp_path="test.pkl", svg_folder="test/", print_option=1):
    start_curve=0
    end_curve=nx9.shape[0]#last curve index

    curve_rec=[]
    for del_curve in range(end_curve):
        if del_curve==0:
            ablation_curve=nx9[1:]
        elif del_curve==(end_curve-1):
            ablation_curve=nx9[:-1]
        else:
            before=nx9[:del_curve,:]
            after=nx9[del_curve+1:,:]
            if nx9[del_curve+1:,-1][0]==nx9[:del_curve,-1][-1]+2:
                #if i delete a curve in a single group
                #move behind curves front 
                after[:,-1]=after[:,-1]-1

            ablation_curve=np.vstack((before,after))

        # check how acc change
        u.dump_item(ablation_curve, temp_path)

        #for debug, not really useful
#         svg_list2, cor2=p.nx92gnn(temp_path, print_option=0, \
#                                   svg_path=svg_folder+"curve %s.svg"%del_curve)
        svg_list2, cor2=p.nx92gnn(temp_path, print_option=0, \
                                  svg_path=svg_folder+"temp.svg")
        curve_tensor2, curve_label2=p.curve_labeling([svg_list2])

        pred_ablation=model(curve_tensor2, curve_label2, [cor2])
    #     pred_brand_ablation=pred_ablation.detach().numpy().argmax(axis=1)[0]
        # true_brand=p.paths2brand([svg_path])[0]#done already
        confidence=pred_ablation.detach().numpy()[0][true_brand]
        curve_rec.append(confidence)
        if print_option==1:
            print(del_curve, confidence)
    return curve_rec


from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, svg2paths, svg2paths2,disvg,wsvg
def svg_list2ablation(name, svg_list, group_features=None, norm=[.002, (0,0,0)], \
            curve_features=None, mode="group", ablation_folder="ablation_svg"):
# def set2svg(pic_idx, svg_list, feature_group=None, feature_group2=None, \
#             curve_feature=None, mode="group"):
    group_idxs, group_ws, group_colors=group_features
    
    ablation_folder=ablation_folder
    pic=svg_list
    svg_path=[]
    color=[]
    feat_w=0.006
    feat_w2=0.0055
    feat_w3=0.0055
    feat_w4=0.005
    feat_w5=0.005
    norm_w=0.002
    norm_w, norm_color=norm
    
    if mode=="group":
        stroke_params=list(np.ones(len(pic))*norm_w)  
        for idx, group in enumerate(pic):
            group_path=0
            
            for idy, curve in enumerate(group):
        #         print(curve)
                c0_x, c0_y, c1_x, c1_y, c2_x, c2_y, c3_x, c3_y=\
                curve
                c0=complex(c0_x, c0_y)
                c1=complex(c1_x, c1_y)
                c2=complex(c2_x, c2_y)
                c3=complex(c3_x, c3_y)
                seg= CubicBezier(c0,c1,c2,c3)
                if group_path==0:
                    group_path=Path(seg)
                else:
                    group_path.append(seg)
                    
            special_group=0
            for group_idx, group_color in zip(group_idxs, group_colors):
#                 group_idx, group_w, group_color= group_feature
                if idx==group_idx:
                    special_group=1
                    color.append(group_color)
                    
            if special_group==0: #it's a normal curve
                color.append(norm_color)
                
            svg_path.append(group_path)
        
        for group_idx, group_w in zip(group_idxs, group_ws):
            stroke_params[group_idx]=group_w 
                      

    if mode=="curve":
        curve_idxs, curve_ws, curve_colors=curve_features
        stroke_params=[]
        prev_idy_sum=0
        for idx, group in enumerate(pic):
            for idy, curve in enumerate(group):
        #         print(curve)
                c0_x, c0_y, c1_x, c1_y, c2_x, c2_y, c3_x, c3_y=\
                curve
                c0=complex(c0_x, c0_y)
                c1=complex(c1_x, c1_y)
                c2=complex(c2_x, c2_y)
                c3=complex(c3_x, c3_y)
                seg= CubicBezier(c0,c1,c2,c3)
                svg_path.append(seg)
                    
                special_group=0
                for group_idx, group_w, group_color in zip(group_idxs, group_ws, group_colors):
                    if idx==group_idx:
                        special_group=1
                        color.append(group_color)
                        stroke_params.append(group_w)
                        
                special_curve=0
                for curve_idx, curve_w, curve_color in zip(curve_idxs, curve_ws, curve_colors):
                    if prev_idy_sum+idy==curve_idx:
#                         print("special curve")
                        special_curve=1
                        
                        if special_group==1: #already a special group, replace
                            color[-1]=curve_color
                            stroke_params[-1]=curve_w
                            
                        if special_group==0:
                            color.append(curve_color)
                            stroke_params.append(curve_w)

                if special_group==0 and special_curve==0: #it's a normal curve
                    color.append(norm_color)
                    stroke_params.append(norm_w)
                
#             svg_path.append(group_path)
            prev_idy_sum+=len(group)
        
        for group_idx, group_w in zip(group_idxs, group_ws):
            stroke_params[group_idx]=group_w 
            
            
    wsvg(svg_path,color,stroke_widths=stroke_params,\
         filename="%s/%s.svg"%(ablation_folder, name))
