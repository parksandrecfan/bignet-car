import preprocess as p
import neural_cuda as n
import evaluation as e
import util as u
import torch
import numpy as np
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, svg2paths, svg2paths2,disvg,wsvg
import os
import argparse



def run(args):
    device="cpu"
    num_brands=args.num_brands
    epoch=args.epoch
    model_folder=args.model_folder
    model_name=args.model_name
    model_path="%s/%s"%(model_folder, model_name)
    # cache=args.cache
    # pkl_cache="test%s.pkl"%cache
    run_train=args.run_train
    run_test=args.run_test
    is_important=args.is_important
    
    dataset_folder=args.dataset_folder
    eval_folder=args.eval_folder
    # look_folder=args.look_folder

    norm_w=.002
    highlight_add=.006
    norm_color=(0,0,0)
    norm=[norm_w, norm_color]
    group_numbers_max=20

    group_thresh=0.001
    eps=1e-20
    confidence=1/num_brands
    if is_important==0:
        confidence=1-confidence

    model=n.bignet2_latent2(device=device, brands=num_brands).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()


    train_test_split_names=u.load_item("dataset/%s/train_test_split_names.pkl"%dataset_folder)

    full_set_path = "dataset/%s"%dataset_folder
    save_folder="%s/%s/%s_epoch/ablation"%(model_folder, eval_folder, epoch)

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

    os.system("mkdir -p %s"%save_folder)
    os.system("mkdir -p %s/train/group"%save_folder)
    os.system("mkdir -p %s/train/group/rec2"%save_folder)
    os.system("mkdir -p %s/train/group/important2"%save_folder)
    os.system("mkdir -p %s/train/group/unimportant2"%save_folder)
    # os.system("mkdir -p %s/train/curve"%save_folder)
    # os.system("mkdir -p %s/train/curve/rec"%save_folder)
    # os.system("mkdir -p %s/train/curve/important2"%save_folder)
    # os.system("mkdir -p %s/train/curve/unimportant2"%save_folder)

    os.system("mkdir -p %s/test/group"%save_folder)
    os.system("mkdir -p %s/test/group/rec2"%save_folder)
    os.system("mkdir -p %s/test/group/important2"%save_folder)
    os.system("mkdir -p %s/test/group/unimportant2"%save_folder)
    # os.system("mkdir -p %s/test/curve"%save_folder)
    # os.system("mkdir -p %s/test/curve/rec2"%save_folder)
    # os.system("mkdir -p %s/test/curve/important2"%save_folder)
    # os.system("mkdir -p %s/test/curve/unimportant2"%save_folder)



    '''
    # if there are enough groups that passes the threshold: 10 is maximum
    # if not enough groups passes the threshold:
        # find the 10th group, see the difference with original
            # if difference is positive, print the difference as threshold (in code minus a eps)
            # if the difference is negative, find the smallest positive as threshold
                #if it's zero, then nothing we can do
    '''
    def highlight_get(names, augs):
        for i, name in enumerate(names[:]):
            true_brand = int(full_array[np.where(full_array[:,2]==name)[0][0],0])

            for aug in augs:
                idx=np.where(short_ids==name)[0][aug]
                svg_list = svg_lists[idx]
                cor = cors[idx]

                curve_tensor, curve_label=p.curve_labeling([svg_list])

                pred=model(curve_tensor, curve_label, [cor])
                group_rec = pred.detach().numpy()[:,true_brand]
                print("name: %s-%s brand: %s"%(name, aug, true_brand))
                group_pkl_path="%s/%s/group/rec2/%s.pkl"%(save_folder, set_dir, name)
                u.dump_item(group_rec, group_pkl_path)

                group_rec=u.load_item(group_pkl_path)
                if is_important==0:
                    group_rec=1-group_rec

                group_numbers=np.sum(group_rec>confidence)
                if group_numbers>group_numbers_max:
                    group_numbers=group_numbers_max
                else: #has to adjust threshold
                    min_thresh=np.sort(group_rec)[-group_numbers_max-1]
                    if min_thresh>confidence:
                        group_thresh=min_thresh+eps #then group_numbers remain 20
                        group_numbers=group_numbers_max
                        print_group_thresh="{:.2e}".format(group_thresh)
                    else: #has to adjust number of groups
                        group_positives=np.where((np.sort(group_rec))>confidence)[0]
                        if group_positives.shape[0]==0:#everything is useless, so number of groups is 0
                            group_numbers=0
                            print_group_thresh="None"
                        else:
                            group_min_positive=group_positives[0]
                            group_thresh=(np.sort(group_rec))[group_min_positive]-eps
                            group_numbers=group_positives.shape[0]
                            print_group_thresh="{:.2e}".format(group_thresh)

                    print("adjusted groups: %s threshold: %s"%(group_numbers, print_group_thresh))

                if group_numbers!=0:
                    group_groups=np.argsort(group_rec)[-group_numbers:]
                    if group_numbers==1:
                        group_ws=[.001+norm_w, norm_w]
                        group_colors=[(250, 0, 0), (250, 200, 200)]
                    else:
                        group_ws=[highlight_add*(i/(group_numbers-1))+norm_w for i in range(group_numbers)]
                        group_colors=[(250,200*(1-i/(group_numbers-1)),200*(1-i/(group_numbers-1))) for i in range(group_numbers)]
                    group_features=[group_groups, group_ws, group_colors]


                    #plot and save
                    if is_important==1:
                        e.svg_list2ablation(name="%s-"%true_brand+name+"-%s"%aug, svg_list=svg_list, group_features=group_features,\
                                            norm=norm, mode="group", ablation_folder="%s/%s/group/important2"%(save_folder, set_dir))

                    else:
                        e.svg_list2ablation(name="%s-"%true_brand+name+"-%s"%aug, svg_list=svg_list, group_features=group_features,\
                                            norm=norm, mode="group", ablation_folder="%s/%s/group/unimportant2"%(save_folder, set_dir))

                        
    if run_train==1:
        set_dir="train"
        X_subset=np.expand_dims(X_train[:,0][y_train<num_brands], 1)
        y_subset=y_train[y_train<num_brands]
        augs=[0]
        names=full_array[X_subset][:,0,2]
        highlight_get(names, augs)
                            
    if run_test==1:
        set_dir="test"
        X_subset=np.expand_dims(X_test[:,0][y_test<num_brands], 1)
        y_subset=y_test[y_test<num_brands]
        augs=[0,1,2,3]
        names=full_array[X_subset][:,0,2]
        highlight_get(names, augs)
        

                        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, default= "model/curve/6brands/random_sample_separateavg_logo_2std-0", help='model_folder')
    parser.add_argument('--model_name', type=str, default="rand_init_car_model 723 epochs_good test.pt", help='model_name')
    parser.add_argument('--dataset_folder', type=str, default= "full_set_separateavg_logo_2std", help='dataset_folder')
    parser.add_argument('--eval_folder', type=str, default= "eval", help='eval_folder')
    # parser.add_argument('--look_folder', type=str, default= "dataset/potracelogo/final", help='--look_folder')
    
    parser.add_argument('--run_train', type=int, default= 1, help='run_train')
    parser.add_argument('--run_test', type=int, default= 1, help='run_test')
    
    parser.add_argument('--is_important', type=int, default=1, help='is_important')    
    parser.add_argument('--epoch', type=int, default=723, help='epoch')
    parser.add_argument('--num_brands', type=int, default=6, help='num_brands')
    
    args = parser.parse_args()
    run(args)
    
# python 13_v2.py \
# --model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
# --model_name "rand_init_car_model 723 epochs_good test.pt" \
# --dataset_folder "full_set_separateavg_logo_2std" \
# --eval_folder "eval" \
# --run_train 0 \
# --run_test 1 \
# --epoch 723 \
# --num_brands 6 \