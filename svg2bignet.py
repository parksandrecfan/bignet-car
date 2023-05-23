import util as u
import preprocess as p
import time
import argparse
def run(args):
    # get the start time
    st = time.time()

    full_set_folder= args.full_set_folder
    dataset_folder = args.dataset_folder
    nx9_folder = args.nx9_folder
    format= args.format
    # format=".svg"
    
    #full set
    full_set_path="%s/%s"%(dataset_folder, full_set_folder)

    train_test_split_names = u.load_item("%s/train_test_split_names.pkl"%(full_set_path))
    labels = list(train_test_split_names['full_array'][:,0].astype(dtype=int))

    svg_lists=[]
    cors=[]
    labels=[]
    # dataset_size=20

    dataset_names = u.get_filelist(dir, "%s/%s"%(dataset_folder, nx9_folder), include=format)

    for name in dataset_names[:]:
        svg_list, cor=p.nx92gnn(nx9_path=name, print_option=0, svg_path="temp/test.svg", normalize=1, rules=None)
        svg_lists.append(svg_list)
        cors.append(cor)



    for name in dataset_names[:]:
        train_test_split_names['label_brand']
        brand=name.split("/")[-1].split("-")[0]
        labels.append(train_test_split_names['label_brand'][brand])



    #save the files!
    u.dump_item(svg_lists,"%s/svg_lists.pkl"%full_set_path)
    u.dump_item(cors,"%s/cors.pkl"%full_set_path)
    u.dump_item(labels,"%s/labels.pkl"%full_set_path)

    curve_tensors, curve_labels=p.curve_labeling(svg_lists)
    u.dump_item(curve_tensors, "%s/curve_tensors.pkl"%full_set_path)
    u.dump_item(curve_labels, "%s/curve_labels.pkl"%full_set_path)

    et = time.time()
    # get the execution time
    elapsed_time = et - st

    print('Execution time:', elapsed_time, 'seconds')

    data_ids = [i.split("/")[-1].split(".")[0] for i in dataset_names]
    u.dump_item(data_ids, "%s/data_ids.pkl"%full_set_path)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_set_folder', type=str, default="full_set", help='full_set_folder')
    parser.add_argument('--dataset_folder', type=str, default="dataset", help='dataset_folder')
    parser.add_argument('--nx9_folder', type=str, default="nx9", help='nx9_folder')
    parser.add_argument('--format', type=str, default=".pkl", help='format')
    
    args = parser.parse_args()
    run(args)
    
# python 9.py --full_set_folder full_set_separateavg_2std --dataset_folder dataset --nx9_folder nx9 --format ".pkl"
# python 9.py --full_set_folder full_set_separateavg_logo_2std --dataset_folder dataset --nx9_folder nx9logo --format ".pkl"