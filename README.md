# NAS-VAD: Neural Architecture Search for Voice Activity Detection
Daniel Rho, Jinhyeok Park, and Jong Hwan Ko


Our code is based on NAS-BOWL (https://github.com/xingchenwan/nasbowl)


# 1. How to run a network search
**1. Prepare speech samples**
- Transform each speech sample into a complex spectrogram with the shape of [channel, n frames, n fft/2 + 1]. In this work, we used [1, n_frames, 257] shaped spectrograms.
- Each spectrogram should be saved to a numpy array beginning with "S." That is, the name should follow the format of "S*.npy", e.g., "S000.npy", "S135ass.npy".
- Extract binary labels (for each preprocessed speech sample and save them with the name of the corresponding speech samples, only "S" being replaced by "L".
- Save all speech samples and labels into a single directory.

**2. Prepare noise samples**
- Follow the exact steps for preparing speech samples, except for the label part.


**3. Running a network search**

**How it works internally**: 
1. The algorithm will randomly sample "n_init" networks from the search space. Each sample network will be trained and its performance will be recorded.
2. Using the initial networks and their performances, the algorithm fits the Gaussian Process.
3. It will sample a total "pool_size" network.
4. Gaussian process choose the top "batch_size" networks from the randomly selected pool of networks.
5. Train and evaluate "batch_size" networks.
6. Using the accumulated data, update the Gaussian process.
7. If the total number of sample networks reaches "max_archs", the algorithm stops; otherwise, go back to 3.

```bash
python3 architecture_search --data_path={your_speech_directory},{your_noise_directory}
```
- "--n_init": controls the initial number of networks for architecture search.
- "--max_archs": the total number of networks to sample during architecture search
- "--batch_size": the number of models per batch.
- "--pool_size": the number of networks within a network pool.
- "--save_path": where to save architecture search results
- "--epochs": how many training epochs

Currently, the number of workers for the dataloader is set at 12.
If you wish to increase the number of workers or if your machine cannot handle that many workers, change the default value of 12 for DARTSTrainer in **darts/arch_trainer.py**.


**How to modify the search space**

- To change the initial channels or the number of layers (cells) for VAD models, modify **darts/darts_config.py**.
- To implement new operations, use **darts/cnn/operations.py**.
- To modify the set of operations for each cell type and the number of operation nodes per cell, go to **darts/darts_objectives.py** and modify the **get_ops** function.

# 2. How to train model

1. you need synthesized spectrogram and label to train model. you should prepare TRAIN, VALID folder. And there should be spectrogram '*_spec.npy' shape of [1, 201, n_frames] and corresponding framewise label '*.npy' (you can refer synthesize_audio.py)

2. There are two types of trainer, one for BDNN like model (trainer.py) and one for Spectro-Temporal Attention(trainer_st_attention.py)

3. you should change the path of train and validation foler by changing CV_TRAIN, TIMIT_TRAIN at trainer.py or trainer_st_attention.py (format should be 'Train_path,Valid_path') 

To run baseline model execute 

```bash
python trainer.py --model 'Search2D' --found 'TIMIT' --mode 'train' --dataset 'TIMIT' --n_mels 80' --save_path "./saved_model"
```

- "--model" : type of model you want to use, and if model is from NAS you have to set genotypes at trainer and use '--found' option to specify your genotypes. (You can refer main part of tainer.py file to set genotypes). BDNN, ACAM, Self Attentive VAD is available.
- "--dataset" : which dataset you will use 
- "--n_mels" : Determine number of mel you want to use
- "--save_path" : path where model will be stored

# 3. How to test model


To test baseline model you should change the path of train and validation foler by changing CV_TEST, TIMIT_TEST at trainer.py or trainer_st_attention.py (format should be '*,Test_path') 

And To test model execute 
```bash
python trainer.py --model 'Search2D' --found 'TIMIT' --mode 'train' --dataset 'TIMIT' --test_dataset 'TIMIT' --n_mels 80' --save_path "./saved_model"
```

- "--model" : type of model you want to use, and if model is from NAS you have to set genotypes at trainer and use '--found' option to specify your genotypes. (You can refer main part of tainer.py file to set genotypes).
- "--dataset" : which dataset used for train  
- "--n_mels" : Determine number of mel you want to use
- "--save_path" : path where model will be loaded
- "--test_dataset" : which dataset you want to test
