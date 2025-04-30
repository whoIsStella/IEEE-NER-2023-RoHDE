# [IEEE NER 2023] Toward Robust High-Density EMG Pattern Recognition useing Generative Adversarial Network and Convolutional Neural Network Implementation

## Abstrct
High-density electromyography (HD EMG)-based Pattern Recognition (PR) has attracted increasing interest in real-time Neural-Machine Interface (NMI) applications because HD EMG can capture neuromuscular information from one temporal and two spatial dimensions, and it does not require anatomically targeted electrode placements. In recent years, deep learning methods such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and hybrid CNN-RNN methods have shown great potential in HD EMG PR. Due to the high-density and multi-channel characteristics of HD EMG, the use of HD EMG-based NMIs in practice may be challenged by the unreliability of HD EMG recordings over time. So far, few studies have investigated the robustness of deep learning methods on HD EMG PR when noises and disturbances such as motion artifacts and bad contacts are present in the HD EMG signals. In this paper, we have developed RoHDE – a Robustdeep learning-based HD EMG PR framework by introducing a Generative Adversarial Network (GAN) that can generate synthetic HD EMG signals to simulate recording conditions affected by disturbances. The generated synthetic HD EMG signals can be utilized to train robust deep learning models against real HD EMG signal disturbances. Experimental results have shown that our proposed RoHDE framework can improve the classification accuracy against disturbances such as contact artifacts and loose contacts from 64% to 99%. To the best of our knowledge, this work is the first to address the intrinsic robustness issue of deep learning-based HD EMG PR.
## Dataset
We are using a private noisy EMG dataset.
The noisy EMG data set consists of the following seven hand and wrist gestures: no movement, wrist supination, wrist pronation, hand close, hand open, wrist flexion, and wrist extension. To evaluate its performance, we experimented with two common disturbances of EMG recordings in this study: Contact Artifacts (CA) and Loose Contacts (LC). The figure below shows two representative trials of HD EMG signals contaminated by LC (Left) and CA (Right), respectively. The LC disturbances were simulated by purposely peeling back the last two rows of an 8×8 HD EMG electrode grid (e.g., channels 8, 16, 24, etc), and placing a towel between those electrodes and the skin. In the CA trials, noise was introduced by tapping a pen on approximately the last 3 dozen electrodes (156-192) at a rate of 4-5 Hz. The exact electrodes affected vary from strike to strike.  
![CA and LC Noise](Images/HD_EMG_LC_CA.png)
## RoHDE Overall Architecture
![Overall Architechture](Images/Overall_Achiecture.png)

## Experiment Result
![Experiment Result](Images/Experiment.png)

## Synthetic Noise EMG signal
![Synthetic CA](Images/Synthetic%20CA.png)
![Synthetic LC](Images/Synthetic%20LC.png)

## Quick Start
### Dependency
```
name: RoHDE
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - cudatoolkit=11.3
  - torchaudio
  - tqdm
  - numpy
  - matplotlib
  - tensorboard
  - pip
pip:
  - fastdtw

```
  
#### Install Dependency
```
conda env create -n RoHDE -f environment.yml
```
```
conda activate RoHDE
```

### GAN Training
```
python WGAN-GP-train.py
```
### EMG-base Gesture Recognition model training
```
python EMG-Classifier.py
```
### Robust  EMG-base Gesture Recognition model training
```
python RoHDE.py
```
