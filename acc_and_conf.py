import util as u
import matplotlib.pyplot as plt
import preprocess as p
# find real accuracy
import evaluation as e
import neural_cuda as n
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import os


def run(args):
    num_brands=args.num_brands
    device = args.device
    epoch = args.epoch

    dataset_name=args.dataset_name

    model_folder = args.model_folder
    model_name=args.model_name
    model_path="%s/%s"%(model_folder, model_name)
    eval_folder=args.eval_folder
    # os.system("mkdir %s/%s"%(model_folder, eval_folder))
    os.system("mkdir -p %s/%s/%s_epoch"%(model_folder, eval_folder, epoch))

    brand_names=["toyota","volkswagen","benz","audi","bmw","hyundai","Nissan","Ford","KIA","Chevy"]
    brand_names=brand_names[:num_brands]

    model=n.bignet2(device=device, brands=args.num_brands).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()


    train_acc=u.load_item("%s/train_acc.pkl"%model_folder)
    test_acc=u.load_item("%s/test_acc.pkl"%model_folder)
    train_loss=u.load_item("%s/train_loss.pkl"%model_folder)
    test_loss=u.load_item("%s/test_loss.pkl"%model_folder)

    plt.title("train/test loss curve")
    plt.plot(train_loss[:])
    ratio = int(len(train_loss)/len(test_loss))
    plt.plot([i for i in test_loss for k in range(ratio)])
    plt.savefig("%s/%s/loss_curve.jpg"%(model_folder, eval_folder))
    plt.close()

    plt.title("train/test accuracy curve")
    plt.plot(train_acc[:])
    plt.plot([i for i in test_acc for k in range(ratio)])
    plt.savefig("%s/%s/acc_curve.jpg"%(model_folder, eval_folder))
    plt.close()

    #load dataset to evaluate
    train_test_split_names=u.load_item("dataset/%s/train_test_split_names.pkl"%dataset_name)

    label_brand = train_test_split_names['label_brand']
    full_array = train_test_split_names['full_array']
    X_train = train_test_split_names['X_train']
    X_test = train_test_split_names['X_test']
    y_train = train_test_split_names['y_train']
    y_test = train_test_split_names['y_test']

    full_set_path = "dataset/%s"%dataset_name
    data_ids=u.load_item("%s/data_ids.pkl"%full_set_path)
    new_labels=u.load_item("%s/labels.pkl"%full_set_path)
    short_ids=np.array([data_id.split("-")[-2] for data_id in data_ids])

    svg_lists=u.load_item("%s/svg_lists.pkl"%full_set_path)
    cors=u.load_item("%s/cors.pkl"%full_set_path)

    # number of brands
    X_train_subset=np.expand_dims(X_train[:,0][y_train<num_brands], 1)
    y_train_subset=y_train[y_train<num_brands]
    X_test_subset=np.expand_dims(X_test[:,0][y_test<num_brands], 1)
    y_test_subset=y_test[y_test<num_brands]

    #train set eval
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

    train_preds=model(train_curve_tensors, train_curve_labels, train_cors)

    train_preds_brand=train_preds.detach().cpu().numpy().argmax(axis=1)
    train_np_labels = train_labels.detach().cpu().numpy()
    train_acc=np.sum(train_preds_brand == train_np_labels)/len(train_np_labels)

    # train confusion
    train_mat=confusion_matrix(train_np_labels, train_preds_brand, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=train_mat,
                                  display_labels=brand_names)
    fig, ax = plt.subplots(figsize=(12,12))
    plt.title("train set confusion matrix",fontsize = 20)
    plt.rcParams.update({'font.size': 12})
    ax.set_ylabel(None,fontsize = 20) # Y label
    ax.set_xlabel(None,fontsize = 20)
    disp.plot(ax=ax, cmap="gray")
    plt.savefig("%s/%s/%s_epoch/train_conf.jpg"%(model_folder, eval_folder, epoch))
    plt.close()

    #test eval
    test_names=full_array[X_test_subset][:,0,2]
    test_chooses=[]
    test_name_chooses=[]
    for i, name in enumerate(test_names):
        # for aug in range(0,1): #test set no augmentation
        options = np.where(short_ids==name)
        for aug in [0]: #80.33%, 82.27%
            choose = options[0][aug]
            if new_labels[np.where(short_ids==name)[0][0]]<num_brands:
                test_chooses.append(choose)
                test_name_chooses.append(name)

    test_labels = [new_labels[choose] for choose in test_chooses]
    test_svg_lists = [svg_lists[choose] for choose in test_chooses]
    test_cors = [cors[choose] for choose in test_chooses]
    test_curve_tensors, test_curve_labels=p.curve_labeling(test_svg_lists)
    test_labels=torch.LongTensor(test_labels).to(device)

    test_preds=model(test_curve_tensors, test_curve_labels, test_cors)

    test_preds_brand=test_preds.detach().cpu().numpy().argmax(axis=1)
    test_np_labels = test_labels.detach().cpu().numpy()
    test_acc=np.sum(test_preds_brand == test_np_labels)/len(test_np_labels)

    test_mat=confusion_matrix(test_np_labels, test_preds_brand, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=test_mat,
                                  display_labels=brand_names)
    fig, ax = plt.subplots(figsize=(12,12))
    plt.title("test set confusion matrix",fontsize = 20)
    plt.rcParams.update({'font.size': 12})
    ax.set_ylabel(None,fontsize = 20) # Y label
    ax.set_xlabel(None,fontsize = 20)

    disp.plot(ax=ax, cmap="gray")
    plt.savefig("%s/%s/%s_epoch/test_conf.jpg"%(model_folder, eval_folder, epoch))
    plt.close()

    #save dictionary
    eval_dir = "%s/%s/%s_epoch/acc_eval.pkl"%(model_folder, eval_folder, epoch)
    eval_dict = {}
    eval_dict["train_names"]=train_name_chooses
    eval_dict["train_preds"]=train_preds_brand
    eval_dict["train_labels"]=train_np_labels
    eval_dict["train_acc"]=train_acc
    eval_dict["train_conf"]=train_mat

    eval_dict["test_names"]=test_name_chooses # image name
    eval_dict["test_preds"]=test_preds_brand # pred brand for each car
    eval_dict["test_labels"]=test_np_labels # label for each image
    eval_dict["test_acc"]=test_acc # accuracy
    eval_dict["test_conf"]=test_mat # confusion matrix

    u.dump_item(eval_dict, eval_dir)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default="full_set_separateavg_logo_2std", help='dataset_name')
    parser.add_argument('--model_folder', type=str, default="model/curve/6brands/random_sample_separateavg_logo_2std-0", help='model_folder')
    parser.add_argument('--model_name', type=str, default="rand_init_car_model 723 epochs_good test.pt", help='model_name')
    parser.add_argument('--eval_folder', type=str, default="eval", help='eval_folder')

    parser.add_argument('--num_brands', type=int, default=6, help='num_brands')
    parser.add_argument('--device', type=int, default=1, help='device')
    parser.add_argument('--epoch', type=int, default=723, help='epoch')

    args = parser.parse_args()
    run(args)
    
# python 11.py \
# --dataset_name "full_set_separateavg_logo_2std" \
# --model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
# --model_name "rand_init_car_model 723 epochs_good test.pt" \
# --eval_folder "eval" \
# --num_brands 6 \
# --device 1 \
# --epoch 723

# python 11.py \
# --dataset_name "full_set_separateavg_2std" \
# --model_folder "model/curve/6brands/random_sample_separateavg_2std-1" \
# --model_name "rand_init_car_model 708 epochs_good test.pt" \
# --eval_folder "eval" \
# --num_brands 6 \
# --device 1 \
# --epoch 708

# python 11.py \
# --dataset_name "full_set_separateavg_logo_2std" \
# --model_folder "model/curve/10brands/random_sample_separateavg_logo_2std-0" \
# --model_name "rand_init_car_model 333 epochs_good test.pt" \
# --eval_folder "eval" \
# --num_brands 10 \
# --device 1 \
# --epoch 333

# python 11.py \
# --dataset_name "full_set_separateavg_2std" \
# --model_folder "model/curve/10brands/random_sample_separateavg_2std-1" \
# --model_name "rand_init_car_model 540 epochs.pt" \
# --eval_folder "eval" \
# --num_brands 10 \
# --device 1 \
# --epoch 540