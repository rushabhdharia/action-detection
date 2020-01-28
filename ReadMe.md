# Action Detection on MERL Shopping Dataset


### MERL Shopping Dataset

The MERL Shopping Dataset contains 106 videos, each of which is a sequence ~2 minutes long. Each video contains several instances of the following 6 actions:
  1. Reach To Shelf     - (reach to shelf)
  2. Retract From Shelf - (retract hand from shelf)
  3. Hand In Shelf      - (extended period with hand in the shelf)
  4. Inspect Product    - (inspect product while holding it in hand)
  5. Inspect Shelf      - (look at shelf while not touching and not reaching for the shelf)
  6. None of the above


## Steps to create environment
1. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
2. chmod u+x Miniconda3-latest-Linux-x86_64.sh 
3. ./Miniconda3-latest-Linux-x86_64.sh 
4. cd miniconda3/
5. conda list
6. source .bashrc
7. conda create -n tf python=3.7 anaconda
8. conda activate tf
9. conda install -c anaconda tensorflow-gpu
10. conda install -c conda-forge opencv


## Steps to run
1. Download and extract MERL Shopping Dataset - ftp://ftp.merl.com/pub/tmarks/MERL_Shopping_Dataset/
2. Download Videos_MERL_Shopping_Dataset.zip and unzip it.
3. Create train, val and test folders inside the unzipped folder for Videos_MERL_Shopping_Dataset.
4. Move 1st 60 videos (1_1_crop to 20_3_crop) to the train folder.
5. Move the next 18 videos (21_1_crop to 26_3_crop) to the val folder
6. Move the remaining 28 videos (27_1_crop to 41_2_crop) to the test folder
7. Then run the following command on the terminal "./convert_video_to_images.sh Videos_MERL_Shopping_Dataset"
8. Repeat the above steps 2-6 similarly for Labels_MERL_Shopping_Dataset and then run Generate_output_pickle_files.ipynb.
9. Activate the conda environment and run ActionRecognition.py on terminal or ActionRecognition.ipynb to run in the jupyter environment 
10. In order to run ResNet Model run Resnet.py


## References
1. Singh, Bharat, et al. "A multi-stream bi-directional recurrent neural network for fine-grained action detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.



## Contributors
1. [Rushabh Dharia](https://github.com/rushabhdharia)
2. [Animesh Sagar](https://github.com/animeshsagar)
 
