# Implementation of HiFi-GAN vocoder.

## Installation guide

As in previous projects the installation process is also pretty simple.

You can use standart Kaggle environment and the following minimal requirements to run the code.

```shell
pip install wandb
pip install librosa
pip install inflect
pip install speechbrain
```

This would be enough to run the inference.

Use the following code for donwloading and running:

```shell
git clone https://github.com/Pe4enkazAMI/tts_project_2.git
cd tts_project_2
python download_weights.py
python test.py -r  ../MyanFi280.pth -t ../tts_project_2/test_data -c ../tts_project/config.json
```

## Details

This is also just an implementation of the HiFi-GAN.
