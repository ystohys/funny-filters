# YSFunnyFilters

Filters:
1. Mouth blurring

## Introduction
This project was born out of the sheer amusement I got from watching my friends use the sad face filter on Snapchat and Instagram (around May 2022). I decided it would be fun to put my knowledge into an actual project so I put a few things together to get this. All these are possible thanks to the developers that build and maintain all packages like OpenCV and PyTorch! And of course, big thanks to Dr Yoshua Bengio of the University of Montreal for his facial keypoint dataset which you can find on Kaggle [here](https://www.kaggle.com/competitions/facial-keypoints-detection/overview). 

** This project is at its infancy because I am buried under a pile of work and it was something I coded just purely for fun. I quite like this particular project, so I will still contribute to it from time to time!


# How to run:
1. Download the environment.yml file and set up the conda environment (preferably on miniconda).
2. Once you have set up and are in your conda environment, use the terminal to navigate to the directory containing these modules (I use 'FunnyFilters' as the directory name).
3. Just run `python3 main.py *name_of_filter*`, with name_of_filter being whichever filter you want (e.g. mouthblur).


# Things to work on:
- Using another architecture for facial keypoint detection, for now it's just a very basic CNN.
- Adding MORE filters! But I foresee this will take quite a while because some filters are really elaborate.