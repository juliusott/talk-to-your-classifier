# talk-to-your-classifier
Implementation for the paper "How to talk to your classifier: Conditional Text Generation from Visual Latent Space"

The ./text_autoencoder/ folder is from [This Github repository](https://github.com/shentianxiao/text-autoencoders.git) with small changes to insert it as a Python module.

### Pre-Training of the DAAE
We refer any user to the [original repository](https://github.com/shentianxiao/text-autoencoders.git) for pretraining the denoising autoencoder
#### Download the data
```
bash download_data.sh
```
#### Execute the training script
```python
python text_autoencoders/train.py --train data/yelp/train.txt --valid data/yelp/valid.txt --model_type aae --lambda_adv 10 --noise 0.3,0,0,0 --save-dir checkpoints/yelp/daae
```

### Generate the captions
They are either generated when the CaptionDataset() class is first called in the training script or can be generated beforehand by executing:
The generated captions can be found in the [this folder](../datasets/) 
```
python caption_dataset.py --caption_model blip-base
```

### Training the Visual Classifier with Text Generation
To recreate the experiments in the paper, execute the following command. 
```
python main.py --dataset cifar10 --caption_model blip-base
```







