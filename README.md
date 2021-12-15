# VAD_NAS

### How to Synthesize audio

1. Prepare speech data and corresponding label. You need sample-wise label, and to run without any modification you need label and audio with same name in same folder. For example, audio_dir/audio1.wav, audio1.npy ...., and you also need noise dataset with audio format.
2. Before running you should split speech data and noise data into train and test. And you should change save path in audio_synthesize.py

### How to run trainer

1. There are three types of trainer, one for using BDNN like window(ACAM_trainer) and one for Spectro-Temporal Attention(ACAM_trainer_st_attention) and the other for 64 frame window(marble_trainer)
2. To run baseline model you can execute like 'python ACAM_trainer.py --gpu '1' --model 'Search2D' --found 'TIMIT' --mode 'train' --dataset 'TIMIT' --n_mels 80'. 
'--model' means type of model you want to use, and if model is from NAS you have to set genotypes at trainer and use '--found' option to specify your genotypes. (You can refer main part of any tainer.py file to set genotypes). 

