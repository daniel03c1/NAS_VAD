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
