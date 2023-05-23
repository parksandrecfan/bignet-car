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

<p>Our approach conducting the car study can be summarized in this flow chart. Note that this page is the implementation of training BIGNet rather than the image processing pipeline. It would allow one to train BIGNet from an SVG dataset (vectorized from a pixel dataset).</p>
<img src="data/flowchart.png" width="1000">

<h2>System requirements</h2>
<p>This project runs on a Linux server.</p>
<p>More hardware information can be found in <b>Hardware Overview.txt</b>.</p>
<p>The required (Python) packages can be found in <b>requirements.txt</b>.</p>

<h2>Visualization Inference Demo Instruction</h2>
<p>1. Download this repo to your linux server. In your terminal:</p>
<p><pre><code>git clone https://github.com/parksandrecfan/bignet-car.git</code></pre></p>
<p>2. Install the python packages in requirements.txt on your linux machine.</p>
<p>Install miniconda <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html">here</a> on your linux server if you haven't. Installing anaconda on a windows computer may also work.</p>
<p>In terminal:</p>
<p><pre><code>
conda create -n BIGNet2
conda activate BIGNet2
pip install -r requirements.txt
</code></pre></p>
<p>3. Directory Initialization. In terminal:</p>
<p><pre><code>
cd biget-car
mkdir dataset
mkdir -p model/curve/6brands/random_sample_logo_2std-0/eval/723_epoch
</code></pre></p>

<p>4. Download dataset <a href="https://drive.google.com/drive/folders/15r5ZrX-pmfTON0Nkw-s3MY-ZsEhez5ez?usp=share_link"><b>"full_set_separateavg_logo_2std"</b></a> and place the folder under "biget-car/dataset".</p>

<p>5. Download the trained BIGNet model <a href="https://drive.google.com/file/d/1RUOH6DCagOAhW18xjYa7030NMhxCXIPW/view?usp=sharing"><b>"rand_init_car_model 723 epochs_good test.pt"</b></a> and place it under "biget-car/curve/6brands/random_sample_logo_2std-0".</p>

<p>6. Calculate confusion matrix. In terminal:</p>
<p><pre><code>
python acc_and_conf.py \
--dataset_name "full_set_separateavg_logo_2std" \
--model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
--model_name "rand_init_car_model 723 epochs_good test.pt" \
--eval_folder "eval" \
--num_brands 6 \
--device 0 \
--epoch 723
</code></pre></p>
<p>The confusion matrix can be found in directory "model/curve/6brands/random_sample_separateavg_logo_2std-0/eval/723_epoch". Train set is "train_conf.jpg" and test set is "test_conf.jpg".</p>

<p>7. Plot tSNE dimension reduction results. In terminal:</p>
<p><pre><code>
python dim_red.py \
--model_folder "model/curve/6brands/random_sample_separateavg_logo_2std-0" \
--dataset_folder "full_set_separateavg_logo_2std" \
--eval_folder "eval" \
--model_name "rand_init_car_model 723 epochs_good test.pt" \
--epoch 723 \
--num_brands 6 \
--device 0 
</code></pre></p>

<p>The confusion matrix and tSNE results show BMW, Benz, and Audi are found to achieve higher recognition rates/clustering compared to other brands. This finding matches the
optimized marketing strategy that luxurious cars value brand
consistency more than economy cars.<b></b></p>
<img src="data/conf_and_tsne.png" width="1000">


<p>8. Visualize the features. In terminal:</p>
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
<p>7. You're done! View the features in "model/curve/6brands/random_sample_separateavg_logo_2std-0/eval/723_epoch/ablation". Redness indicates the attention.</p>
<p>* Train set, brand-relevant: <b>train/group/important2</b>.</p>
<p>* Train set, brand-irrelevant: <b>train/group/unimportant2</b>.</p>
<p>* Test set, brand-relevant: <b>test/group/important2</b>.</p>
<p>* Test set, brand-irrelevant: <b>test/group/unimportant2</b>.</p>
<p>Here are a few visualization results. It is obvious that BIGNet captures luxury segments’ well-distinguishable car parts including grille, headlights and fog lights, while there are much fewer geometric clues on affordable cars (Toyota) that it has to rely on logo detection:</p>
<img src="data/cam.png" width="1000">

<p><b>-------------------More Detailed Documentation-------------------</b></p>
<p>This instruction will go through how to train BIGNet on 6-brand car images that has logos. For variations of 10 brands and logo removal, and the comparison with pixel CNN (simCLR+gradCAM) results, please email me, Sean Chen, at yuhsuan2@andrew.cmu.edu and I'll be happy to provide more support!</p>
<h3>SVG/nx9 Dataset</h3>
<p>The dataset format that would be used down the pipeline is nx9, and it can be downloaded <a href="https://drive.google.com/file/d/1EhZq4pBseJuNEIUJ-KXauxGjvlZU9cOC/view?usp=share_link"><b>here</b></a>(has to decompress). Each pickle file is a nx9 numpy matrix that represents a car's front view like the image below. The one-to-one correspondence SVG files can be found <a href="https://drive.google.com/file/d/1zU_wg6gt2tVKp9Nh09x_-dqSyGOWK-K0/view?usp=share_link"><b>here</b></a>(has to decompress). The files that share the same names represent the same car image. 0,1,2,3 at the end of each filename indicates the 4 augmented images of the same car. The SVG files are provided to give users a glance of the images, and is not actually being used down the pipeline. (In addition, one could easily verify the correspondence using the nx92svg and svg2nx9 functions in "preprocess.py" to convert between the two data format).</p>
<p><b>The following pipeline places the decompressed nx9 folder under "dataset/" directory.</b></p>

<h3>Preprocess nx9 data for BIGNet</h3>
<p>The process to convert the SVG (nx9 numpy) dataset to a BIGNet-friendly format is done in "svg2bignet.py". <b>First, download "<a href="https://drive.google.com/file/d/1AKov83vd_zc_PVQo2Lg3yVdfkmuxcvPx/view?usp=share_link"><b>train_test_split_names.pkl</b></a>" and place it in "full_set_folder" directory</b> (here, we use "full_set_separateavg_logo_2std" as the name of the "full_set_folder"). Then, run this example command to type in the terminal:</p> 

<pre>
  <code>
python svg2bignet.py \
--full_set_folder \
full_set_separateavg_logo_2std \
--dataset_folder dataset \
--nx9_folder nx9 \
--format ".pkl”
  </code>
</pre>

<p>The command would return 6 files, including "data_ids.pkl", "svg_lists.pkl", "cors.pkl", "labels.pkl", "curve_tensors.pkl", and "curve_labels.pkl" also in "full_set_folder".</p>

The paths are:
nx9 dataset: dataset/nx9
BIGNet formatted dataset: dataset/full_set_separateavg_logo_2std/*.pkl


you may also just skip this preprocessing section by doing step 4 in the demo section above to download the processed data directly, and place it in a folder named "dataset".


<h3>Training</h3>
<p>To train with 6 brands cars with logos using the first GPU of your device, download <a href="https://drive.google.com/file/d/1hwrOMvQraFToymJT_aeyu4tIsIgNdAqm/view?usp=share_link"><b>"full_dataset_separateavg_logo_2std.txt"</a> and place it in the main directory. Also, make a folder named "model" to store your trained BIGNet model</b>. Then, run python train.py in terminal. For example, Type the following command in terminal:</p>
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
<p>The pretrained model of "rand_init_car_model 723 epochs_good test.pt" can be downloaded <a href="https://drive.google.com/file/d/1RUOH6DCagOAhW18xjYa7030NMhxCXIPW/view?usp=share_link"><b>here</b></a></p> for one to test.</p>

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
