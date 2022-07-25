# Inception V3 for TV Human Interactions dataset Research Workshop CSAI

## About
This project aims to classify human interactions in video frames with a new training regime. Training video data usually takes a long time because of large dataset size. Frames close to each other may not provide enough different information. Hence, this project experimented training frames in intervals. The network gets as inputs images extracted every n frames from videos. We varied n as a part of our experiment. For example, we may train the model with frames 1,4,7,10,etc when n is 3. Inception V3 is trained on the Oxford TV Human Interactions dataset. Transfer Learning is applied (weights trained on Imagenet). For more details, please check the file "individual_paper_KhoiNQ_2039045_resit.pdf"

<p align="center">
  <img  src="https://github.com/khoinguyen19k8/Inception_v3_TV_Human_Interactions_CSAI/blob/master/plots/AveragePlot2.png"></p>

*The experiment results*



## Installation
Git is required to download and install the repo. You can open Terminal (for Linux and Mac) or cmd (for Windows) and follow these commands:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/khoinguyen19k8/Inception_v3_TV_Human_Interactions_CSAI.git
```

## Dependencies
The network was build with Keras while using the TensorFlow backend.  `scikit-learn` was used as a supplementary package for doing a train-validation split. Additionally. OpenCV was needed to handle videos and images. Matplotlib was used to plot the results.
```sh
$ sudo pip install -U scikit-learn
$ sudo pip install --upgrade tensorflow
$ sudo pip install opencv-python
$ sudo pip install matplotlib
```

## References
This work is based on the following two papers:
1. Patron-Perez, Alonso, et al. "High Five: Recognising human interactions in TV shows." BMVC, 2010. [[link]](http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html)
2. Stergiou,  A., & Poppe,  R.   (2018).   Understanding human-human interactions: a survey.CoRR,abs/1808.00022. Re-trieved from http://arxiv.org/abs/1808.00022

If you use this repository for your work, you can cite it as:
```sh
@misc{Khoi2021,
    author={Nguyen, Quang Khoi},
    title={Frame strides experiment - Inception V3 - TV Human Interactions dataset},
    year= 2021, 
    url = {https://github.com/khoinguyen19k8/Inception_v3_TV_Human_Interactions_CSAI}
}
```

## License

MIT

## Acknowledgments

I am grateful for GPU and disk space support from my teammate, Jos Prinsen.

## Contact
Nguyen Quang Khoi

k.q.nguyen@tilburguniversity.edu

Any queries or suggestions are much appreciated!
