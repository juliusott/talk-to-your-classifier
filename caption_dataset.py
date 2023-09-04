from torch.utils.data import Dataset
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import json
from typing import Dict
from text_autoencoders.batchify import get_batch, get_batches

cifar_100_coarse_labels = {"0": "aquatic_mammals", "1": "fish", "2": "flowers", "3": "food_containers", "4": "fruit_and_vegetables", "5": "household_electrical_devices", "6": "household_furniture", "7": "insects",
                            "8": "large_carnivores", "9": "large_man-made_outdoor_things", "10": "large_natural_outdoor_scenes", "11": "large_omnivores_and_herbivores", "12": "medium_mammals",
                            "13": "non-insect_invertebrates", "14": "people", "15": "reptiles", "16": "small_mammals",  "17": "trees", "18": "vehicles_1", "19": "vehicles_2"}

labels_to_words_mnist = {"0" : "digit zero" ,
                   "1" : "digit one" ,
                   "2" : "digit two",
                   "3" : "digit three",
                   "4" : "digit four",
                   "5" : "digit five",
                   "6" : "digit six",
                   "7" : "digit seven",
                   "8" : "digit eight",
                   "9" : "digit nine"}

labels_to_words_cifar10 = {"0" : "airplane" ,
                   "1" : "automobile" ,
                   "2" : "bird",
                   "3" : "cat",
                   "4" : "deer",
                   "5" : "dog",
                   "6" : "frog",
                   "7" : "horse",
                   "8" : "ship",
                   "9" : "truck"}


def label_names_cifar100():
    with open('/home/pplcount/users/ottj/data/cifar100_label_names.txt') as f:
        lines = f.readlines()

    f.close()

    lines = [line.replace("\n", "").split(": ") for line in lines]
    labels_to_words_cifar100 = dict(zip([line[0] for line in lines], [line[1] for line in lines]))

    return labels_to_words_cifar100

labels_to_words_cifar100 = label_names_cifar100()

class TextCoder(object):
    def __init__(self, checkpoint="git-base") -> None:
        if "git" in checkpoint:
            self.checkpoint = f"microsoft/{checkpoint}"
            self.checkpoint_name= checkpoint
            self.processor = AutoProcessor.from_pretrained(self.checkpoint, cache_dir="/home/pplcount/users/ottj/")

            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, cache_dir="/home/pplcount/users/ottj/")
        elif "blip" in checkpoint:
            model_size = checkpoint.replace("blip-", "")
            self.checkpoint = f"Salesforce/blip-image-captioning-{model_size}"
            self.checkpoint_name= checkpoint
            self.processor = BlipProcessor.from_pretrained(self.checkpoint, cache_dir="/home/pplcount/users/ottj/")
            self.model = BlipForConditionalGeneration.from_pretrained(self.checkpoint, cache_dir="/home/pplcount/users/ottj/")



    def get_word_idx(self, sent: str, word: str):
        return sent.split(" ").index(word)

    def get_caption(self, image):
        inputs = self.processor(images=image,return_tensors="pt")
        pixel_values = inputs.pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=20)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption


class LLMDataset(Dataset):
    def __init__(self, dataset , transform=None, mode="train", dataset_name="cifar10", caption_model="git-base", vocab= None) -> None:
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        self.text_encoder = TextCoder(checkpoint=caption_model)
        self.text_embeddings = None
        self.vocab = vocab
        self.init_text_embeddings()

    def init_text_embeddings(self):
        json_file_path = f"./datasets/captions_{self.dataset_name}_{self.mode}_{self.text_encoder.checkpoint_name}.json"
        if os.path.isfile(json_file_path):
            with open(json_file_path)as f:
                self.captions = json.load(f)
            print("loaded captions")
            
        else:
            caption_dict = dict()
            self.text_embeddings = []
            with torch.no_grad():
                idx = 0
                for sample, _ in tqdm(self.dataset):
                    caption = self.text_encoder.get_caption(sample)
                    caption_dict[str(idx)] = caption
                    idx += 1
                    #embedding = np.expand_dims(self.text_encoder.text_to_embedding(caption).cpu().numpy(), axis=0)
                    #self.text_embeddings.append(embedding)


            
            with open(json_file_path, "w") as f:
                json.dump(caption_dict, f)
            
            self.captions = caption_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset.__getitem__(idx)


        if self.transform:
            sample = self.transform(sample)

        caption = self.captions[str(idx)]
        input_batch, _  = get_batches([caption.split()], self.vocab, batch_size=1)

        inputs, targets = input_batch[0]
        caption_input = inputs.squeeze()
        caption_target = targets.squeeze()

        return sample, label, caption_input, caption_target


if __name__ == "__main__":
    from argparse import Namespace
    from torchvision import datasets

    def set_loader(opt):

        train_dataset = datasets.CIFAR10(root=opt.data_folder,                                         
                                            download=False)
        train_dataset = LLMDataset(dataset=train_dataset, labels_to_words=labels_to_words_cifar10, transform=None, mode="train", dataset_name="cifar10", caption_model=opt.caption_model)

        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        train=False)

        val_dataset = LLMDataset(dataset=val_dataset, labels_to_words=labels_to_words_cifar10, transform=None, mode="val", dataset_name="cifar10", caption_model=opt.caption_model)

        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                            download=True)

        train_dataset = LLMDataset(dataset=train_dataset, labels_to_words=labels_to_words_cifar10, transform=None, mode="train", dataset_name="cifar100", caption_model=opt.caption_model)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False)
        
        val_dataset = LLMDataset(dataset=val_dataset, labels_to_words=labels_to_words_cifar10, transform=None, mode="val", dataset_name="cifar100", caption_model=opt.caption_model)

    opt = Namespace()
    opt.dataset = "cifar10"
    opt.data_folder = '/home/pplcount/users/ottj/data/'
    opt.caption_model = "blip-base"
    for caption_model in ["git-base", "git-large"]:
        opt.caption_model = caption_model
        print("settings: ", opt)
        set_loader(opt)


    