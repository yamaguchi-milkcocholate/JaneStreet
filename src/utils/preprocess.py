from typing import *
from pathlib import Path
from tqdm.notebook import tqdm
import os
import sys
import gc
import pickle
import random
import numpy as np
import pandas as pd
import albumentations as A
import cv2
import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import StratifiedKFold


class Transform:
    def __init__(self, aug_kwargs: Dict):
        self.transform = A.Compose([getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()])

    def __call__(self, image):
        image = self.transform(image=image)["image"]
        return image


class ChestXRayDataset(Dataset):
    def __init__(self, filepaths: List[str], labels: List[int], label_smoothing: float = 0., mixup_prob: float = 0., image_transform: Transform = None, oversample: bool = False):
        self.filepaths = filepaths
        self.labels = labels
        if oversample:
            self.__oversample()
        
        self.label_smoothing = label_smoothing
        self.mixup_prob = mixup_prob
        self.image_transform = image_transform
        
    def __len__(self) -> int:
        return len(self.filepaths)
    
    def __getitem__(self, idx: int):
        img, label = self.__get_single_example(idx=idx)
        img, label = self.__mixup(img=img, label=label)
        
        label_logit = torch.tensor([1 - label, label], dtype=torch.float32)
        
        return img, label_logit
    
    def __get_single_example(self, idx: int) -> Tuple[torch.Tensor, float]:
        img = cv2.imread(self.filepaths[idx])
        if self.image_transform:
            img = self.image_transform(img)
        img = torch.tensor(np.transpose(img, (2, 0, 1)).astype(np.float32))
        
        label = self.labels[idx]
        if label == 0:
            return img, float(label) + self.label_smoothing
        else:
            return img, float(label) - self.label_smoothing
    
    def __mixup(self, img: torch.Tensor, label: float):
        if np.random.uniform() < self.mixup_prob:
            pair_idx = np.random.randint(0, len(self.filepaths))
            prob = np.random.uniform()
            pair_img, pair_label = self.__get_single_example(idx=pair_idx)
            img = img * self.mixup_prob + pair_img * (1 - self.mixup_prob)
            label = label * self.mixup_prob + pair_label * (1 - self.mixup_prob)
        return img, label
    
    def __oversample(self):
        labels = np.array(self.labels)
        n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
        
        sample_class = 1 if n_pos < n_neg else 0
        n_sample = np.abs(n_pos - n_neg)
        
        print(f'#Positive: {n_pos}, #Negative: {n_neg}  | over sample: Class{sample_class} {n_sample}')
        
        population = np.where(labels == sample_class)[0]
        random.seed(111)
        sample_idx = random.choices(population.tolist(), k=n_sample)
        
        self.filepaths += np.array(self.filepaths)[sample_idx].tolist()
        self.labels += np.array(self.labels)[sample_idx].tolist()


class StratifiedKFoldWrapper:
    def __init__(self, datadir: Path, n_splits: int, shuffle: bool, seed: int, 
                 label_smoothing: float = 0., mixup_prob: float = 0., aug_kwargs: Dict = {}, debug: bool = False, oversample: bool = True):
        self.datadir = datadir
        fl = self.__load_filelist()
        fp = self.__load_filepath()
        
        self.datalist = pd.merge(fl, fp, how='left', on='Image Index')[['Image Index', 'Normal', 'Patient ID', 'File Path', 'No Finding Rate']]
        self.datalist = self.datalist.rename(columns={'Image Index': 'image_index', 'Normal': 'normal', 'Patient ID': 'patient_id', 'File Path': 'filepath', 'No Finding Rate': 'no_finding_rate'})
        self.datalist = self.datalist.query("(no_finding_rate == 0) | (no_finding_rate == 1)").reset_index(drop=True)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        self.split_idxs = list(skf.split(self.datalist['filepath'].values, self.datalist['normal'].values))
        self.__i = -1
        
        self.label_smoothing = label_smoothing
        self.mixup_prob = mixup_prob
        self.image_transform = Transform(aug_kwargs=aug_kwargs)

        self.debug = debug
        self.oversample = oversample
        
    def __load_filelist(self):
        df = pd.read_csv(self.datadir / 'Data_Entry_2017.csv')
        df1 = df.groupby('Patient ID').first().reset_index()
        df2 = df.groupby('Patient ID')['Finding Labels'].agg(lambda x: (x == 'No Finding').sum() / len(x)).reset_index()
        df2.columns = ['Patient ID', 'No Finding Rate']
        df = pd.merge(df1, df2, how='inner', on='Patient ID')[['Patient ID', 'Image Index', 'No Finding Rate']]
        df['Normal'] = (df['No Finding Rate'] == 1).astype('int8')
        
        del df1, df2
        gc.collect()
        
        return df
        
    def __load_filepath(self):
        records = {'Image Index': [], 'File Path': []}
        
        for dn in [tmp for tmp in sorted(os.listdir(self.datadir)) if 'image' in tmp]:
            for fn in sorted(os.listdir(self.datadir / dn / 'images')):
                records['Image Index'] += [fn]
                records['File Path'] += [str(self.datadir / dn / 'images' / fn)]
                
        return pd.DataFrame(records)
    
    def __iter__(self):
        self.__i = -1
        return self

    def __next__(self):
        self.__i += 1
        if self.__i < 0 or len(self.split_idxs) <= self.__i:
            raise StopIteration()
        return self.generate_datasets()
    
    def __len__(self) -> int:
        return len(self.split_idxs)
    
    def __getitem__(self, idx):
        if idx < 0 or len(self.split_idxs) <= idx:
            raise ValueError()
        self.__i = idx
        return self.generate_datasets()
        
    def generate_datasets(self) -> Tuple[ChestXRayDataset, ChestXRayDataset]:
        train_idx, valid_idx = self.split_idxs[self.__i]
        if self.debug:
            n_sample = 1000
            train_idx = train_idx[random.sample(list(range(len(train_idx))), n_sample)]
            valid_idx = valid_idx[random.sample(list(range(len(valid_idx))), n_sample)]
        
        train_ds = ChestXRayDataset(
            filepaths=self.datalist['filepath'].values[train_idx].tolist(),
            labels=self.datalist['normal'].values[train_idx].tolist(),
            label_smoothing=self.label_smoothing,
            mixup_prob=self.mixup_prob,
            image_transform=self.image_transform,
            oversample=self.oversample
        )
        valid_ds = ChestXRayDataset(
            filepaths=self.datalist['filepath'].values[valid_idx].tolist(),
            labels=self.datalist['normal'].values[valid_idx].tolist(),
        )
        
        return train_ds, valid_ds