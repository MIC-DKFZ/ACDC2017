# ACDC 2017 Challenge contribution (segmentation only)
This is the code that was used for our participation in 2017's Automatic Cardiac Diagnosis Challenge. We were happy to 
achieve the first place in the segmentation part of the contest with the highest Dice scores achieved for all classes 
and time steps (ED/ES). An extended version of our paper will be published in the MICCAI STACOM proceedings. 
A preliminary version of our paper can be found here: 
https://arxiv.org/abs/1707.00587

The challenge leaderboard is available here:
http://acdc.creatis.insa-lyon.fr/#phase/5966175c6a3c770dff4cc4fb (login required, unfortunately)

### Windows is not supported!
Batchgenerators (which we use in training) do not work with Windows yet, so you won't be able to run the training if you are on Windows. Sorry for the inconvenience

## How to use
This code was cleaned up and made usable for external users, but is still what the authors would like to call 'messy'.
As you can imagine, a challenge deadline will lead to very rapid code development in the very end which ultimately 
produces less user friendly code. We did our best to improve the usability so that you should be able to reproduce our 
results easily.

We use two different architectures for this challenge, a 2D and a 3D UNet. The way the code looks for these networks 
is very different because they were initially taken from different projects. UNet2D has simple training and 
architecture files whereas UNet3D is object oriented. You don't need to deal with this if you just wish to train the 
networks and use our results.

### Prerequisites
Our code is only compatible with python 2.7. Running it with python 3 will need modifications.

We depend on some python packages which need to be installed by the user (may be incomplete):
* theano
* lasagne
* SimpleITK
* sklearn
* numpy
* batchgenerators (get it here: https://github.com/MIC-DKFZ/batchgenerators)

You need to have downloaded at least the training set of the ACDC challenge.

### Training set preprocessing

```
python dataset_utils.py -i INPUT_FOLDER -out2d OUTPUT_FOLDER_FOR_2D_DATA -out3d OUTPUT_FOLDER_FOR_3D_DATA
```
INPUT_FOLDER: Folder to which you downloaded and extracted the training data

OUTPUT_FOLDER_FOR_2D_DATA: Where to save training data for 2d network

OUTPUT_FOLDER_FOR_3D_DATA: Where to save training data for 3d network

### Training the networks
First go into the `UNet2D_config.py` and `UNet2D_config.py` and adapt all the paths to match your file system and the 
download locations of training and test sets.

Training the networks requires a GPU with at least 12 GB of VRAM. On a Pascal Titan X, a 2D UNet trains in < 1d and a 
3D UNet takes approx. 2 days.

```
python run_training_2D.py -f FOLD -c CONFIG_FILE_2D
python run_training_3D.py -f FOLD -c CONFIG_FILE_3D

```

FOLD is the fold id of the cross-validation (0, 1, 2, 3, 4) and CONFIG_FILE_2D/CONFIG_FILE_3D is the path to 
UNet2D_config.py/UNet3D_config.py. You need to train all 5 folds for 2D and 3D UNet to be able to predict the test set 
(test results are obtained by using the 10 resulting networks as ensemble).

You can run the validation of the cross-validation like this:
```
python run_validation_2D.py -f FOLD -c CONFIG_FILE_2D
python run_validation_3D.py -f FOLD -c CONFIG_FILE_3D

```

### Test set prediction
Change into the test_set direcory, then run
```
python predict_test_2D_net.py -f FOLD -c CONFIG_FILE_2D
python predict_test_3D_net.py -f FOLD -c CONFIG_FILE_3D

```

for all five folds and both networks.

Finally, consolidate the predictions of the networks with
```
python create_final_submission_files.py -c2d CONFIG_FILE_2D -c3d CONFIG_FILE_3D -o OUTPUT_FOLDER
```

The predicted test set will be saved in OUTPUT_FOLDER

## Contact
If you wish to contact us regarding problems, questions or suggestions, please write an email to either 
Fabian (f.isensee@dkfz.de) or Paul (p.jaeger@dkfz.de)
