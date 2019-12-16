# Action Detection on MERL Shopping Dataset


### MERL Shopping Dataset

The MERL Shopping Dataset contains 106 videos, each of which is a sequence ~2 minutes long. Each video contains several instances of the following **5 actions:**


  1. **Reach To Shelf**      (reach to shelf)

  2. **Retract From Shelf**  (retract hand from shelf)

  3. **Hand In Shelf**       (extended period with hand in the shelf)

  4. **Inspect Product**     (inspect product while holding it in hand)

  5. **Inspect Shelf**       (look at shelf while not touching and not reaching for the shelf)


## Steps to run on Cloud Instance
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod u+x Miniconda3-latest-Linux-x86_64.sh 

./Miniconda3-latest-Linux-x86_64.sh 

conda

cd miniconda3/

conda list

source .bashrc

conda create -n tf python=3.7 anaconda

conda activate tf

conda install -c anaconda tensorflow-gpu

conda install -c conda-forge opencv


## References

1. Singh, Bharat, et al. "A multi-stream bi-directional recurrent neural network for fine-grained action detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

2. Moghaddam, Mohammad Mahdi Kazemi, Ehsan Abbasnejad, and Javen Shi. "Follow the Attention: Combining Partial Pose and Object Motion for Fine-Grained Action Detection." arXiv preprint arXiv:1905.04430 (2019).

## Contributors
1. [Rushabh Dharia](https://github.com/rushabhdharia)
2. [Animesh Sagar](https://github.com/animeshsagar)
3. [Devanshi Mittal](https://github.com/mittaldevanshi)
 
