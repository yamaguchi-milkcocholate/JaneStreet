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

import torchvision
assert torch.__version__.startswith("1.7")
from detectron2.structures import BoxMode


class Transform:
    def __init__(self, aug_kwargs: Dict):
        self.transform = A.Compose([getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()])

    def __call__(self, image):
        image = self.transform(image=image)["image"]
        return image


class ChestXRayDataset(Dataset):
    def __init__(self, filepaths: List[str], labels: List[int], label_smoothing: float = 0., mixup_prob: float = 0., 
                 image_transform: Transform = None, oversample: bool = False, downsample: bool = False):
        self.filepaths = filepaths
        self.labels = labels

        assert not (oversample and downsample)
        if oversample:
            self.__oversample()
        
        elif downsample:
            self.__downsample()
        
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

    def __downsample(self):
        labels = np.array(self.labels)
        n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
        
        sample_class = 1 if n_pos > n_neg else 0
        n_sample = np.minimum(n_pos, n_neg)

        print(f'#Positive: {n_pos}, #Negative: {n_neg}  | down sample: Class{sample_class} {n_sample}')

        population = np.where(labels == sample_class)[0]
        random.seed(111)
        sample_idx = random.choices(population.tolist(), k=n_sample)

        remain_idx = np.where(labels != sample_class)[0].tolist()

        self.filepaths = np.array(self.filepaths)[sample_idx + remain_idx].tolist()
        self.labels = np.array(self.labels)[sample_idx + remain_idx].tolist()


class StratifiedKFoldWrapper:
    def __init__(self, datadir: Path, n_splits: int, shuffle: bool, seed: int, 
                 label_smoothing: float = 0., mixup_prob: float = 0., aug_kwargs: Dict = {}, debug: bool = False, oversample: bool = False, downsample: bool = False):
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
        self.downsample = downsample
        
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
            n_sample = 50
            train_idx = train_idx[random.sample(list(range(len(train_idx))), n_sample)]
            valid_idx = valid_idx[random.sample(list(range(len(valid_idx))), n_sample)]
        
        train_ds = ChestXRayDataset(
            filepaths=self.datalist['filepath'].values[train_idx].tolist(),
            labels=self.datalist['normal'].values[train_idx].tolist(),
            label_smoothing=self.label_smoothing,
            mixup_prob=self.mixup_prob,
            image_transform=self.image_transform,
            oversample=self.oversample,
            downsample=self.downsample,
        )
        valid_ds = ChestXRayDataset(
            filepaths=self.datalist['filepath'].values[valid_idx].tolist(),
            labels=self.datalist['normal'].values[valid_idx].tolist(),
        )
        
        return train_ds, valid_ds


def get_vinbigdata_dicts(
    imgdir: Path,
    train_df: pd.DataFrame,
    train_data_type: str = "original",
    use_cache: bool = True,
    debug: bool = True,
    target_indices: Optional[np.ndarray] = None
    ):
    debug_str = f"_debug{int(debug)}"
    train_data_type_str = f"_{train_data_type}"
    cache_path = Path(".") / f"dataset_dicts_cache{train_data_type_str}{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        train_meta = pd.read_csv(imgdir / "train_meta.csv")
        if debug:
            train_meta = train_meta.iloc[:500]  # For debug....

        # Load 1 image to get image size.
        image_id = train_meta.loc[0, "image_id"]
        image_path = str(imgdir / "train" / f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, train_meta_row in tqdm(train_meta.iterrows(), total=len(train_meta)):
            record = {}

            image_id, height, width = train_meta_row.values
            filename = str(imgdir / "train" / f"{image_id}.png")
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            objs = []
            for index2, row in train_df.query("image_id == @image_id").iterrows():
                # print(row)
                # print(row["class_name"])
                # class_name = row["class_name"]
                class_id = row["class_id"]
                if class_id == 14:
                    # It is "No finding"
                    # This annotator does not find anything, skip.
                    pass
                else:
                    # bbox_original = [int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])]
                    h_ratio = resized_height / height
                    w_ratio = resized_width / width
                    bbox_resized = [
                        int(row["x_min"]) * w_ratio,
                        int(row["y_min"]) * h_ratio,
                        int(row["x_max"]) * w_ratio,
                        int(row["y_max"]) * h_ratio,
                    ]
                    obj = {
                        "bbox": bbox_resized,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_id,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]
    return dataset_dicts


def get_vinbigdata_dicts_test(imgdir: Path, test_meta: pd.DataFrame, use_cache: bool = True, debug: bool = True):
    debug_str = f"_debug{int(debug)}"
    cache_path = Path(".") / f"dataset_dicts_cache_test{debug_str}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        # test_meta = pd.read_csv(imgdir / "test_meta.csv")
        if debug:
            test_meta = test_meta.iloc[:500]  # For debug....

        # Load 1 image to get image size.
        image_id = test_meta.loc[0, "image_id"]
        image_path = str(imgdir / "test" / f"{image_id}.png")
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, test_meta_row in tqdm(test_meta.iterrows(), total=len(test_meta)):
            record = {}

            image_id, height, width = test_meta_row.values
            filename = str(imgdir / "test" / f"{image_id}.png")
            record["file_name"] = filename
            # record["image_id"] = index
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            # objs = []
            # record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts