import neural_cuda as n
import preprocess as p
import evaluation as e
import util as u
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import argparse


def run(args):
    num_brands=args.num_brands
    device=args.device
    model_folder=args.model_folder
    eval_folder=args.eval_folder
    epoch=args.epoch
    dataset_folder=args.dataset_folder
    model_name=args.model_name

    model_path="%s/%s"%(model_folder, model_name)
    train_test_split_names=u.load_item("dataset/%s/train_test_split_names.pkl"%dataset_folder)
    full_set_path = "dataset/%s"%dataset_folder
    save_folder="%s/%s/%s_epoch"%(model_folder, eval_folder, epoch)

    label_brand = train_test_split_names['label_brand']
    full_array = train_test_split_names['full_array']
    X_train = train_test_split_names['X_train']
    X_test = train_test_split_names['X_test']
    y_train = train_test_split_names['y_train']
    y_test = train_test_split_names['y_test']


    data_ids=u.load_item("%s/data_ids.pkl"%full_set_path)
    new_labels=u.load_item("%s/labels.pkl"%full_set_path)
    short_ids=np.array([data_id.split("-")[-2] for data_id in data_ids])

    svg_lists=u.load_item("%s/svg_lists.pkl"%full_set_path)
    cors=u.load_item("%s/cors.pkl"%full_set_path)

    model=n.bignet2_latent1(device=device, brands=num_brands).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    #color code
    color_code=[
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [0,0,0],
        [0.5,0.5,0.5],
        [1,0.5,0],
        [0.5,1,0]
    ]

    patch0 = mpatches.Patch(color=color_code[0], label="toyota")
    patch1 = mpatches.Patch(color=color_code[1], label="volksagen")
    patch2 = mpatches.Patch(color=color_code[2], label="benz")
    patch3 = mpatches.Patch(color=color_code[3], label="audi")
    patch4 = mpatches.Patch(color=color_code[4], label="bmw")
    patch5 = mpatches.Patch(color=color_code[5], label="hyundai")
    patch6 = mpatches.Patch(color=color_code[6], label="nissan")
    patch7 = mpatches.Patch(color=color_code[7], label="ford")
    patch8 = mpatches.Patch(color=color_code[8], label="kia")
    patch9 = mpatches.Patch(color=color_code[9], label="chevy")

    patches=[patch0,\
    patch1,\
    patch2,\
    patch3,\
    patch4,\
    patch5,\
    patch6,\
    patch7,\
    patch8,\
    patch9]

    color_code = color_code[:num_brands]
    patches = patches[:num_brands]

    #train set
    X_train_subset=np.expand_dims(X_train[:,0][y_train<num_brands], 1)
    y_train_subset=y_train[y_train<num_brands]

    train_names=full_array[X_train_subset][:,0,2]
    train_chooses=[]
    train_name_chooses=[]
    for i, name in enumerate(train_names):
        # for aug in range(0,1): #test set no augmentation
        options = np.where(short_ids==name)
        aug =0 # should be hflip, but not selecting the correct ones
        choose = options[0][aug]
        if new_labels[np.where(short_ids==name)[0][0]]<num_brands:
            train_chooses.append(choose)
            train_name_chooses.append(name)

    train_labels = [new_labels[choose] for choose in train_chooses]
    train_svg_lists = [svg_lists[choose] for choose in train_chooses]
    train_cors = [cors[choose] for choose in train_chooses]
    train_curve_tensors, train_curve_labels=p.curve_labeling(train_svg_lists)
    train_labels=torch.LongTensor(train_labels).to(device)

    #get vector
    train_preds=model(train_curve_tensors, train_curve_labels, train_cors)
    train_latent_vec = train_preds[0].detach().cpu().numpy()

    train_tsne_vec=e.vec2plot(train_latent_vec, mode="tsne", dimension=2)
    train_tsne_tanh_vec=e.vec2plot(np.tanh(train_latent_vec), mode="tsne", dimension=2)
    train_pca_vec=e.vec2plot(train_latent_vec, mode="pca", dimension=2)
    train_pca_tanh_vec=e.vec2plot(np.array(torch.tanh(train_preds[0]).detach().cpu(), dtype=np.float64), mode="pca", dimension=2)

    train_colors=[]
    for label in train_labels:
        train_colors.append(color_code[label])

    plt.figure(figsize=(10,10))
    plt.scatter(train_tsne_vec[:,0],train_tsne_vec[:,1], color=train_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/train_tsne.png"%save_folder)
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(train_tsne_tanh_vec[:,0],train_tsne_tanh_vec[:,1], color=train_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/train_tsne_tanh.png"%save_folder)
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(train_pca_vec[:,0],train_pca_vec[:,1], color=train_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/train_pca.png"%save_folder)
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(train_pca_tanh_vec[:,0],train_pca_tanh_vec[:,1], color=train_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/train_pca_tanh.png"%save_folder)
    plt.close()

    #test set
    X_test_subset=np.expand_dims(X_test[:,0][y_test<num_brands], 1)
    y_test_subset=y_test[y_test<num_brands]

    test_names=full_array[X_test_subset][:,0,2]
    test_chooses=[]
    test_name_chooses=[]
    for i, name in enumerate(test_names):
        # for aug in range(0,1): #test set no augmentation
        options = np.where(short_ids==name)
        for aug in [0,1,2,3]:
            choose = options[0][aug]
            if new_labels[np.where(short_ids==name)[0][0]]<num_brands:
                test_chooses.append(choose)
                test_name_chooses.append(name)

    test_labels = [new_labels[choose] for choose in test_chooses]
    test_svg_lists = [svg_lists[choose] for choose in test_chooses]
    test_cors = [cors[choose] for choose in test_chooses]
    test_curve_tensors, test_curve_labels=p.curve_labeling(test_svg_lists)
    test_labels=torch.LongTensor(test_labels).to(device)

    #get vector
    test_preds=model(test_curve_tensors, test_curve_labels, test_cors)
    test_latent_vec = test_preds[0].detach().cpu().numpy()

    test_tsne_vec=e.vec2plot(test_latent_vec, mode="tsne", dimension=2)
    test_tsne_tanh_vec=e.vec2plot(np.tanh(test_latent_vec), mode="tsne", dimension=2)
    test_pca_vec=e.vec2plot(test_latent_vec, mode="pca", dimension=2)
    test_pca_tanh_vec=e.vec2plot(np.array(torch.tanh(test_preds[0]).detach().cpu(), dtype=np.float64), mode="pca", dimension=2)

    test_colors=[]
    for label in test_labels:
        test_colors.append(color_code[label])

    plt.figure(figsize=(10,10))
    plt.scatter(test_tsne_vec[:,0],test_tsne_vec[:,1], color=test_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/test_tsne.png"%save_folder)
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(test_tsne_tanh_vec[:,0],test_tsne_tanh_vec[:,1], color=test_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/test_tsne_tanh.png"%save_folder)
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(test_pca_vec[:,0],test_pca_vec[:,1], color=test_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/test_pca.png"%save_folder)
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(test_pca_tanh_vec[:,0],test_pca_tanh_vec[:,1], color=test_colors, s=15)
    plt.legend(handles=patches, fontsize=12)
    plt.savefig("%s/test_pca_tanh.png"%save_folder)
    plt.close()

    #save dictionary
    dim_red_dict={}
    dim_red_dict["train_latent_vec"]=train_latent_vec
    dim_red_dict["train_pca_vec"]=train_pca_vec
    dim_red_dict["train_pca_tanh_vec"]=train_pca_tanh_vec
    dim_red_dict["train_tsne_vec"]=train_tsne_vec
    dim_red_dict["train_tsne_tanh_vec"]=train_tsne_tanh_vec

    dim_red_dict["test_latent_vec"]=test_latent_vec
    dim_red_dict["test_pca_vec"]=test_pca_vec
    dim_red_dict["test_pca_tanh_vec"]=test_pca_tanh_vec
    dim_red_dict["test_tsne_vec"]=test_tsne_vec
    dim_red_dict["test_tsne_tanh_vec"]=test_tsne_tanh_vec

    u.dump_item(dim_red_dict, "%s/dim_rec.pkl"%save_folder)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default= "model/curve/6brands/random_sample_separateavg_logo_2std-0", help='--model_folder')
    parser.add_argument('--dataset_folder', type=str, default= "full_set_separateavg_logo_2std", help='--dataset_folder')
    parser.add_argument('--eval_folder', type=str, default="eval", help='--eval_folder')    
    parser.add_argument('--model_name', type=str, default="rand_init_car_model 723 epochs_good test.pt", help='--model_name')
    
    parser.add_argument('--epoch', type=int, default=723, help='--epoch')
    parser.add_argument('--num_brands', type=int, default=6, help='--num_brands')
    parser.add_argument('--device', type=int, default=1, help='--device')
    
    args = parser.parse_args()
    run(args)


# python 12.py \
# --model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
# --dataset_folder "full_set_separateavg_logo_2std" \
# --eval_folder "eval" \
# --model_name "rand_init_car_model 723 epochs_good test.pt" \
# --epoch 723 \
# --num_brands 6 \
# --device 1
