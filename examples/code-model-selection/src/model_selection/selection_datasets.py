import datetime as dt
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle
from collections import defaultdict
import random
import glob

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from tqdm import tqdm

import utils

import datasets

import task_configuration

dataset_objs     = {}
test_transforms  = {}
train_transforms = {}




##################
# Oxford IIIT Dogs
##################

class OxfordPets(Dataset):
	"""`Oxford Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
	Args:
		root (string): Root directory of dataset where directory
			``omniglot-py`` exists.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		download (bool, optional): If true, downloads the dataset tar files from the internet and
			puts it in root directory. If the tar files are already downloaded, they are not
			downloaded again.
	"""
	folder = 'oxford_pets'

	def __init__(self,
				 root,
				 train=True,
				 transform=None,
				 loader=default_loader):

		self.root = os.path.join(os.path.expanduser(root), self.folder)
		self.train = train
		self.transform = transform
		self.loader = loader
		self._load_metadata()

	def __getitem__(self, idx):

		sample = self.data.iloc[idx]
		path = os.path.join(self.root, 'images', sample.img_id) + '.jpg'

		target = sample.class_id - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)

		return img, target, idx

	def _load_metadata(self):
		if self.train:
			train_file = os.path.join(self.root, 'annotations', 'trainval.txt')
			self.data = pd.read_csv(train_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])
		else:
			test_file = os.path.join(self.root, 'annotations', 'test.txt')
			self.data = pd.read_csv(test_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])

	def __len__(self):
		return len(self.data)

dataset_objs['oxford_pets'] = OxfordPets

train_transforms['oxford_pets'] = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomResizedCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms['oxford_pets'] = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


##################
# CodeXGLUE Defect Detection (Devign)
##################
class Devign(Dataset):
	def __init__(self, path, train=False, transform=None, feature_extractor=None):
		self.train = train
		self.split = 'train' if self.train else 'test'
		self.fe = feature_extractor
		self.ds = datasets.load_dataset('code_x_glue_cc_defect_detection', download_mode="reuse_dataset_if_exists")

	def __getitem__(self, idx):
		func = self.ds[self.split]['func'][idx]
		target = 1 if self.ds[self.split]['target'][idx] == True else 0
		return func, target, idx

	def __len__(self):
		return len(self.ds[self.split])

dataset_objs['codexglue_defect_detection'] = Devign

####################
# Dataset Loader
####################



def construct_dataset(dataset:str, path:str, train:bool=False, **kwdargs) -> torch.utils.data.Dataset:
	# transform = (train_transforms[dataset] if train else test_transforms[dataset])
	# transform = test_transforms[dataset] # Note: for training, use the above line. We're using the train set as the probe set, so use test transform
	return dataset_objs[dataset](path, train, transform=None, **kwdargs)

def get_dataset_path(dataset:str) -> str:
	return f'./data/{dataset}/'


class ClassMapCache:
	""" Constructs and stores a cache of which instances map to which classes for each datset. """

	def __init__(self, dataset:str, train:bool):
		self.dataset = dataset
		self.train = train

		if not os.path.exists(self.cache_path):
			self.construct_cache()
		else:
			with open(self.cache_path, 'rb') as f:
				self.idx_to_class, self.class_to_idx = pickle.load(f)


	def construct_cache(self):
		print(f'Constructing class map for {self.dataset}...')
		dataset    = construct_dataset(self.dataset, get_dataset_path(self.dataset), self.train)
		dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

		self.idx_to_class = []
		self.class_to_idx = defaultdict(list)

		idx = 0

		for batch in tqdm(dataloader):
			y = batch[1]
			single_class = (y.ndim == 1)

			for _cls in y:
				if single_class:
					_cls = _cls.item()
				
				self.idx_to_class.append(_cls)
				
				if single_class:
					self.class_to_idx[_cls].append(idx)
				
				idx += 1
		
		self.class_to_idx = dict(self.class_to_idx)

		utils.make_dirs(self.cache_path)
		with open(self.cache_path, 'wb') as f:
			pickle.dump((self.idx_to_class, self.class_to_idx), f)



	@property
	def cache_path(self):
		return f'./cache/class_map/{self.dataset}_{"train" if self.train else "test"}.pkl'


class DatasetCache(torch.utils.data.Dataset):
	""" Constructs and stores a cache for the dataset post-transform. """

	def __init__(self, dataset:str, train:bool):
		self.dataset = dataset
		self.train = train

		self.cache_folder = os.path.split(self.cache_path(0))[0]
		
		if not os.path.exists(self.cache_path(0)):
			os.makedirs(self.cache_folder, exist_ok=True)
			self.construct_cache()
		
		self.length = len(glob.glob(self.glob_path()))
		self.class_map = ClassMapCache(dataset, train)
		
		super().__init__()

	def cache_path(self, idx:int) -> str:
		return f'./cache/datasets/{self.dataset}/{"train" if self.train else "test"}_{idx}.json'
		
	def glob_path(self) -> str:
		return f'./cache/datasets/{self.dataset}/{"train" if self.train else "test"}_*'

	def construct_cache(self):
		print(f'Constructing dataset cache for {self.dataset}...')
		dataset    = construct_dataset(self.dataset, get_dataset_path(self.dataset), self.train)
		dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

		idx = 0

		for batch in tqdm(dataloader):
			x = batch[0]

			# self.x_cache = x

			for i in range(len(x)):
				# np.save(self.cache_path(idx), x[i].numpy().astype(np.float16))
				with open(self.cache_path(idx), 'w') as f:
					json.dump(x[i], f)
				idx += 1
	
	def __getitem__(self, idx:int) -> tuple:
		# x = torch.from_numpy(np.load(self.cache_path(idx)).astype(np.float32))
		with open(self.cache_path(idx), 'r') as f:
			x = json.load(f)
		y = self.class_map.idx_to_class[idx]
		return x, y

	def __len__(self):
		return self.length


class BalancedClassSampler(torch.utils.data.DataLoader):
	""" Samples from a dataloader such that there's an equal number of instances per class. """

	def __init__(self, dataset:str, batch_size:int, instances_per_class:int, train:bool=True, **kwdargs):
		num_classes = task_configuration.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train)
		map_cache = ClassMapCache(dataset, train)

		sampler_list = []

		for _, v in map_cache.class_to_idx.items():
			random.shuffle(v)
		
		for _ in range(instances_per_class):
			for i in range(num_classes):
				if i in map_cache.class_to_idx:
					idx_list = map_cache.class_to_idx[i]
					
					if len(idx_list) > 0:
						sampler_list.append(idx_list.pop())
		
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)


class FixedBudgetSampler(torch.utils.data.DataLoader):
	""" Samples from a dataloader such that there's a fixed number of samples. Classes are distributed evenly. """

	def __init__(self, dataset:str, batch_size:int, probe_size:int, train:bool=True, min_instances_per_class:int=2, **kwdargs):
		num_classes = task_configuration.num_classes[dataset]
		dataset_obj = DatasetCache(dataset, train)
		map_cache = ClassMapCache(dataset, train)

		# VOC is multiclass so just sample a random subset
		if dataset == 'voc2007':
			samples = list(range(len(dataset_obj)))
			random.shuffle(samples)

			super().__init__(dataset_obj, batch_size, sampler=samples[:probe_size], **kwdargs)
			return

		sampler_list = []
		last_len = None

		for _, v in map_cache.class_to_idx.items():
			random.shuffle(v)
		
		class_indices = list(range(num_classes))
		class_indices = [i for i in class_indices if i in map_cache.class_to_idx] # Ensure that i exists

		# Whether or not to subsample the classes to meet the min_instances and probe_size quotas 
		if num_classes * min_instances_per_class > probe_size:
			# Randomly shuffle the classes so if we need to subsample the classes, it's random.
			random.shuffle(class_indices)
			# Select a subset of the classes to evaluate on.
			class_indices = class_indices[:probe_size // min_instances_per_class]
		
		# Updated the list of samples (sampler_list) each iteration with 1 image for each class
		# We stop when we're finished or there's a class we didn't add an image for (i.e., out of images).
		while last_len != len(sampler_list) and len(sampler_list) < probe_size:
			# This is to ensure we don't infinitely loop if we run out of images
			last_len = len(sampler_list)

			for i in class_indices:
				idx_list = map_cache.class_to_idx[i]
				
				# If we still have images left of this class
				if len(idx_list) > 0:
					# Add it to the list of samples
					sampler_list.append(idx_list.pop())
				
				if len(sampler_list) >= probe_size:
					break
		
		super().__init__(dataset_obj, batch_size, sampler=sampler_list, **kwdargs)
		

if __name__ == '__main__':
	FixedBudgetSampler('voc2007', 128, 500, train=True)
