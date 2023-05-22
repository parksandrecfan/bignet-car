<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>BIGNet - Car</h1>
<h4><a href="https://drive.google.com/file/d/1qFROn8uz7wG6HjcUkMC8Rdc0atgCuA22/view?usp=share_link">[Paper]</a></h4>

<p>This page is the implementation of BIGNet in the car case study. Phone case study implementation can be found <a href="https://github.com/parksandrecfan/bignet-phone"><b>here</b>.</p>
<h2>Project summary</h2>
<p>Identifying and codifying brand-related aesthetic features for produc redesign is essential yet challenging, even for humans. This project demonstrates a deep learning, data-driven way to automatically learn brand-related features through SVG-based supervised learning, using brand identification graph neural network (<b>BIGNet</b>), a hierarichal graph neural network.</p>

<p>Our approach conducting the car study can be summarized in this flow chart. Note that this page is the implementation of training BIGNet rather than the image processing process. It would allow one to train BIGNet from an SVG dataset.</p>
<img src="data/flowchart.png" width="1000">

<h2>System requirements</h2>
<p>This project runs on a Linux server.</p>
<p>More hardware information can be found in <b>Hardware Overview.txt</b>.</p>
<p>The required (Python) packages can be found in <b>requirements.txt</b>.</p>

<h2>Instruction</h2>
<p>This instruction will go through how to train BIGNet on 6-brand car images that has logos. For variations of 10 brands and logo removal, and the comparison with pixel CNN (simCLR+gradCAM) results, please email me, Sean Chen, at yuhsuan2@andrew.cmu.edu and I'll be happy to provide more support!</p>
<h3>SVG/nx9 Dataset</h3>
<p>The dataset can be downloaded <a href="this is nx9 folder"><b>here</b></a>. Each pickle file is a nx9 numpy matrix that represents a car's front view like the image below. The one-to-one correspondence SVG files can be found <a href="this is potrace/final"><b>here</b></a>. (One could also verify the correspondence using the nx92svg and svg2nx9 functions in "preprocess.py").</p>

<h3>Preprocess nx9 data for BIGNet</h3>
The process to converge the SVG (nx9 numpy) dataset to a BIGNet-friendly format is done in "SVG2BIGNet.py". Place "train_test_split_names.pkl" in "full_set_path", and it would return 5 files, including "svg_lists.pkl", "cors.pkl", "labels.pkl", "curve_tensors.pkl", and "curve_labels.pkl" also in "full_set_path". This the the example code to type in the terminal:</p>
<pre>
  <code>
python SVG2BIGNet.py \
--full_set_folder \
full_set_separateavg_logo_2std \
--dataset_folder dataset \
--nx9_folder nx9 \
--format ".pkl”
  </code>
</pre>

If you wish to skip this step, the 5 files can be downloaded <a href=""><b>here</b></a>.


<h3>Training</h3>
<p>To train with 6 brands cars with logos using the first GPU of your device, make sure <b>"full_dataset_separateavg_logo.txt" is in the directory and that a folder to store your trained BIGNet model exists</b>(if not, create an empty folder). Then, run python train.py in terminal. For example, Type the following command in terminal:</p>
<pre>
  <code>
python train.py \
--dataset_names_path "full_dataset_separateavg_logo_2std.txt" \
--full_set_folder "full_set_separateavg_logo" \
--model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
--brands 6 \
--device 0
  </code>
</pre>
Your trained model, and train/test loss and accuracy history will be stored in the model_folder directory. The paths are:
<p>dataset: "dataset/full_set_separateavg_logo_2std"</p>
<p>models: "model/curve/6brands/random_sample_separateavg_logo_2std-0/*.pt</p>

<h3>Accuracy and confusion matrix</h3>
To know the accuracy and confusion matrix of a desired BIGNet model, run "acc_and_conf.py". For example, type this in terminal:
<pre>
  <code>
python acc_and_conf.py \
--dataset_name "full_set_separateavg_logo_2std" \
--model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
--model_name "rand_init_car_model 723 epochs_good test.pt" \
--eval_folder "eval" \
--num_brands 6 \
--device 0 \
--epoch 723
  </code>
</pre>
<p>In this snippet, it calculates the confusion matrix of a given dataset and model, and save the train/test accuracy curve, loss curve and confusion matrix plots in directory.</p>
<p>The paths are:</p>
<p>model: "model/curve/6brands/random_sample_separateavg_logo_2std-0/rand_init_car_model 723 epochs_good test.pt"</p>
<p>dataset: "dataset/full_set_separateavg_logo_2std"</p>
<p>evaluation folder: "model/curve/6brands/random_sample_separateavg_logo_2std-0/eval/723_epoch"</p>
<p>The pretrained model of "rand_init_car_model 723 epochs_good test.pt" can be downloaded <a href=""><b>here</b></a></p> for one to test.</p>

<h3>Dimension Reduction</h3>
<p>Dimension reduction of 2D/3D PCA/tSNE is done in "dim_red.py". For example, type this in terminal:</p>
<pre>
  <code>
    python dim_red.py \
    --model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
    --dataset_folder "full_set_separateavg_logo_2std" \
    --eval_folder "eval" \
    --model_name "rand_init_car_model 723 epochs_good test.pt" \
    --epoch 723 \
    --num_brands 6 \
    --device 0 
  </code>
</pre>
<p>In this snippet, it outputs the train/test sets' 2D tsne/PCA plots in evaluation folder. The paths are identical to the <b>Accuracy and confusion matrix</b> section.</p>

<h3>cam-ablation studies</h3>
<p>Visualization of brand-related features are implemented in "cam_ablation.py". For example, type this in terminal:</p>
<pre>
  <code>
python cam_ablation.py \
--model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
--model_name "rand_init_car_model 723 epochs_good test.pt" \
--dataset_folder "full_set_separateavg_logo_2std" \
--eval_folder "eval" \
--run_train 1 \
--run_test 1 \
--is_important 1 \
--epoch 723 \
--num_brands 6 
  </code>
</pre>
<p>In this snippet, it outputs the brand-relevant/irrelevant features of the desired model on each sample in dataset. The paths are:</p>
<p>model: "model/curve/6brands/random_sample_separateavg_logo_2std-0/rand_init_car_model 723 epochs_good test.pt"</p>
<p>dataset: "dataset/full_set_separateavg_logo_2std"</p>
<p>ablation folder: "model/curve/6brands/random_sample_separateavg_logo_2std-0/eval/723_epoch/ablation"</p>
<p>brand-relevant images: ablation folder + train/test + group/important2</p>
<p>brand-irrelevant images: ablation folder + train/test + group/unimportant2</p>

<p>For any questions implementing, feel free to email Sean Chen as yuhsuan2@andrew.cmu.edu</p>

</body>
</html>
