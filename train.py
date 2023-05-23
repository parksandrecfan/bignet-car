import util as u
import preprocess as p
import numpy as np
import neural_cuda as n
import torch
import torch.nn as nn
from torch.optim import Adam
import random
import time

from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import argparse
def run(args):
    dataset_names_path = args.dataset_names_path
    dataset_folder = args.dataset_folder
    full_set_folder = args.full_set_folder
    model_folder = args.model_folder
    pretrained_model_path="%s/%s"%(args.model_folder, args.pretrained_model_path)
    pretrained_epochs=args.pretrained_epochs
    brands=args.brands
    device=args.device
    epochs = args.epochs
    lr=args.lr
    batch_size=args.batch_size

    start_time=time.time()
    
    dataset_names=u.load_item(dataset_names_path)
    full_set_path="%s/%s"%(dataset_folder, full_set_folder)
    svg_lists=u.load_item("%s/svg_lists.pkl"%full_set_path)
    cors=u.load_item("%s/cors.pkl"%full_set_path)
    labels=u.load_item("%s/labels.pkl"%full_set_path)
    data_ids=u.load_item("%s/data_ids.pkl"%full_set_path)
    train_test_split_names = u.load_item("%s/train_test_split_names.pkl"%full_set_path)

    label_brand = train_test_split_names['label_brand']
    full_array = train_test_split_names['full_array']
    X_train = train_test_split_names['X_train']
    X_test = train_test_split_names['X_test']
    y_train = train_test_split_names['y_train']
    y_test = train_test_split_names['y_test']

    # limit only first 6 brands
    X_train=np.expand_dims(X_train[:,0][y_train<brands], 1)
    y_train=y_train[y_train<brands]

    X_test=np.expand_dims(X_test[:,0][y_test<brands], 1)
    y_test=y_test[y_test<brands]



    random_oversampler = RandomOverSampler(sampling_strategy='auto', random_state=0)
    X_train_resampled, y_train_resampled = random_oversampler.fit_resample(X_train, y_train)    
    X_test_resampled, y_test_resampled = random_oversampler.fit_resample(X_test, y_test) 

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]


    loss_func = nn.CrossEntropyLoss()

    train_loss_path = "%s/train_loss.pkl"%model_folder
    train_acc_path = "%s/train_acc.pkl"%model_folder
    test_loss_path = "%s/test_loss.pkl"%model_folder
    test_acc_path = "%s/test_acc.pkl"%model_folder

    brand_d=n.bignet2(device=device, brands=brands, cor_init_rand=1).to(device)

    #to continue training
    if pretrained_epochs != 0:
        brand_d.load_state_dict(torch.load(pretrained_model_path))
        train_acc_record=u.load_item("%s/train_acc.pkl"%model_folder)
        test_acc_record=u.load_item("%s/test_acc.pkl"%model_folder)
        train_loss_record=u.load_item("%s/train_loss.pkl"%model_folder)
        test_loss_record=u.load_item("%s/test_loss.pkl"%model_folder)
        ratio = int(len(train_acc_record)/len(test_acc_record))
        train_acc_record=  train_acc_record[:pretrained_epochs*ratio]
        test_acc_record=  test_acc_record[:pretrained_epochs]
        train_loss_record=  train_loss_record[:pretrained_epochs*ratio]
        test_loss_record=  test_loss_record[:pretrained_epochs]
        
    #from scratch
    else:
        # when starting from scratch
        train_loss_record = []
        train_acc_record = []
        test_loss_record = []
        test_acc_record = []


    optim = Adam(brand_d.parameters(), lr=lr)

    train_iters = y_train_resampled.shape[0]//batch_size
    test_iters = y_test_resampled.shape[0]//batch_size
    # iters = 1
    short_ids=np.array([data_id.split("-")[-2] for data_id in data_ids])

    #pre-make test set
    test_idxs = X_test_resampled
    names = full_array[list(test_idxs[:,0])][:,2]

    test_chooses=[]
    for i, name in enumerate(names):
        # for aug in range(0,1): #test set no augmentation
        options = np.where(short_ids==name)
        aug = np.random.randint(0,4)
        choose = options[0][aug]
        test_chooses.append(choose)

    test_labels = [labels[choose] for choose in test_chooses]
    test_svg_lists = [svg_lists[choose] for choose in test_chooses]
    test_cors = [cors[choose] for choose in test_chooses]
    test_curve_tensor, test_curve_label=p.curve_labeling(test_svg_lists)
    test_labels=torch.LongTensor(test_labels).to(device)

    for epoch in range(epochs):
        brand_d.train()
        X_train_resampled, y_train_resampled = unison_shuffled_copies(X_train_resampled, y_train_resampled)
        for train_iter in range(train_iters):
            train_idxs = X_train_resampled[batch_size*train_iter : batch_size*(train_iter+1)]
            names = full_array[list(train_idxs[:,0])][:,2]

            chooses=[]
            for i, name in enumerate(names):
                options = np.where(short_ids==name)
                aug = np.random.randint(0,4)
                choose = options[0][aug]
                chooses.append(choose)

            #batch infos
            train_labels = [labels[choose] for choose in chooses]
            train_svg_lists = [svg_lists[choose] for choose in chooses]
            train_cors = [cors[choose] for choose in chooses]

            train_curve_tensor, train_curve_label=p.curve_labeling(train_svg_lists)

            train_preds=brand_d(train_curve_tensor, train_curve_label, train_cors)
            train_labels=torch.LongTensor(train_labels).to(device)
    #         print(labels_batch)
            train_loss=loss_func(train_preds, train_labels)
            train_acc=accuracy_score(train_labels.cpu(), train_preds.max(axis=1).indices.cpu(), normalize=True)
            train_loss_record.append(train_loss.item())
            train_acc_record.append(train_acc)

            optim.zero_grad()
            train_loss.backward(retain_graph=True)
            optim.step()


            #if this model has the smallest loss, and has run more than 50 epochs, then save it
            if (epoch+pretrained_epochs)>50 and min(train_loss_record)==train_loss_record[-1]:
                print("new low train loss: %s"%train_loss.item())
                torch.save(brand_d.state_dict(), "%s/rand_init_car_model %s epochs %s iter.pt"%(model_folder, epoch+pretrained_epochs, train_iter))

        u.dump_item(train_loss_record, train_loss_path)  

        if epoch%20==0:
            torch.save(brand_d.state_dict(), "%s/rand_init_car_model %s epochs.pt"%(model_folder, epoch+pretrained_epochs))
            end_time=time.time()
            print("20 epochs time usage: %s sec"%round(end_time-start_time, 2))
            start_time=time.time()



        brand_d.eval()
        test_preds=brand_d(test_curve_tensor, test_curve_label, test_cors)
        test_loss=loss_func(test_preds, test_labels)
        test_acc=accuracy_score(test_labels.cpu(), test_preds.max(axis=1).indices.cpu(), normalize=True)
        test_loss_record.append(test_loss.item())
        test_acc_record.append(test_acc)
        
        if (epoch+pretrained_epochs)>50 and max(test_acc_record)==test_acc_record[-1]:
            print("new high test acc: %s"%test_acc.item())
            torch.save(brand_d.state_dict(), "%s/rand_init_car_model %s epochs_good test.pt"%(model_folder, epoch+pretrained_epochs))
            
        u.dump_item(train_loss_record, train_loss_path)
        u.dump_item(train_acc_record, train_acc_path)
        u.dump_item(test_loss_record, test_loss_path)
        u.dump_item(test_acc_record, test_acc_path)

        if epoch%10==0:
            print("%s epoch train loss: %s, acc: %s, test loss: %s, acc: %s"\
                  %(epoch, round(train_loss.item(),3), round(train_acc.item(),3), round(test_loss.item(),3), round(test_acc.item(),3)))


    torch.save(brand_d.state_dict(), "%s/rand_init_car_model %s epochs.pt"%(model_folder, epoch+pretrained_epochs))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_names_path', type=str, default="full_dataset.txt", help='dataset_names_path')
    parser.add_argument('--dataset_folder', type=str, default="dataset", help='dataset_folder')
    parser.add_argument('--full_set_folder', type=str, default="full_set", help='full_set_folder')
    parser.add_argument('--model_folder', type=str, default="model/curve/6brands/random_sample", help='model_folder')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained_model_path')
    parser.add_argument('--pretrained_epochs', type=int, default=0, help='pretrained_epochs')
    parser.add_argument('--brands', type=int, default=6, help='brands')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--epochs', type=int, default=3000, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    args = parser.parse_args()
    run(args)
    
# python 10.py \
# --dataset_names_path "full_dataset_separateavg_logo_2std.txt" \
# --full_set_folder "full_set_separateavg_logo_2std" \
# --model_folder "model/curve/10brands/random_sample_separateavg_logo_2std-0" \
# --pretrained_model_path "rand_init_car_model 235 epochs_good test.pt" \
# --pretrained_epochs 235 \
# --brands 10 \
# --device 2 \
# --lr 0.001