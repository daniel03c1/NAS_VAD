# NAS-VAD: Neural Architecture Search for Voice Activity Detection
Daniel Rho, Jinhyeok Park, and Jong Hwan Ko (https://arxiv.org/abs/2201.09032)

Our code is based on NAS-BOWL (https://github.com/xingchenwan/nasbowl)

# 0. How to instantiate our model
```bash
from darts.cnn.genotypes import Genotype
from darts.cnn.model import NetworkVADv2

# config
channel = 40
n_cells = 4
genotype = Genotype(normal=[('SE_0.25', 0), ('MBConv_3x3_x2', 1),
                            ('zero', 2), ('SE_0.25', 0),
                            ('MBConv_5x5_x4', 3), ('MBConv_5x5_x4', 2),
                            ('sep_conv_5x5', 2), ('MBConv_5x5_x2', 1)],
                    normal_concat=range(2, 6),
                    reduce=[('MBConv_3x3_x4', 1), ('MBConv_3x3_x4', 0),
                            ('MBConv_5x5_x2', 2), ('MHA2D_2', 0),
                            ('MHA2D_4', 2), ('FFN2D_1', 1),
                            ('GLU2D_5', 4), ('MBConv_5x5_x2', 2)],
                    reduce_concat=range(2, 6))
use_second = True
n_frames = 7
n_mels = 80

net = NetworkVADv2(channel, n_cells, genotype, use_second,
                   height=n_frames, width=n_mels)
```


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

1. you need a synthesized spectrogram and label to train a model. you should prepare TRAIN, VALID, TEST folder. And place [1, 201, n_frames] shaped spectrogram  "\*_spec.npy" . And place corresponding framewise label "\*.npy" in same folder. (you can refer synthesize_audio.py to synthesize audio)

2. you should change the path of train and validation folder by changing CV_TRAIN, TIMIT_TRAIN at trainer.py (format should be 'Train_path,Valid_path') 

3. To train baseline model, execute following line. 

```bash
python trainer.py --model 'Darts2D' --found 'TIMIT' --mode 'train' --dataset 'TIMIT' --n_mels 80' --save_path "./saved_model"
```

- "--model" : Types of the model you want to use, and if the model is from NAS you have to set genotypes at 'get_model' at trainer.py. And use '--found' option to specify your genotypes. (You can refer main part of tainer.py file to set genotypes). BDNN, ACAM, Self Attentive VAD, Spectro-Temporal Attention is additionally available.
- "--dataset" : Dataset to be used for training.
- "--n_mels" : The number of mel you want to use. (default : 64)
- "--save_path" : A path where a model will be stored. Just enter the folder name. It will save a model automatically.

# 3. How to test model

To test baseline model, you should change the path of train and validation foler by changing CV_TEST, TIMIT_TEST at trainer.py (format should be "*,Test_path") 

And To test model, execute 
```bash
python trainer.py --model 'Darts2D' --found 'TIMIT' --mode 'test' --dataset 'TIMIT' --test_dataset 'TIMIT' --n_mels 80' --save_path "./saved_model"
```
- "--model" : Types of the model you want to use, and if the model is from NAS you have to set genotypes at 'get_model' at trainer.py. And use '--found' option to specify your genotypes. (You can refer main part of tainer.py file to set genotypes). BDNN, ACAM, Self Attentive VAD, Spectro-Temporal Attention is additionally available.
- "--dataset" : Dataset to be used for training.  
- "--n_mels" : The number of mel you want to use. (default : 64)
- "--save_path" : A path where a model will be stored. Just enter the folder name. It will save a model automatically.
- "--test_dataset" : Dataset to be used for testing.
