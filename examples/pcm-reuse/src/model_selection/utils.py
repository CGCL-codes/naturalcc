import torch
import torchvision

import random
import numpy as np

import os
import csv

import task_configuration
from feature_extractor import FeatureExtractor

def get_source_model_path(architecture:str, source_dataset:str) -> str:
	return f'./models/{architecture}/{architecture}_{source_dataset}.pth'

def get_transfer_model_path(architecture:str, source_dataset:str, target_dataset:str) -> str:
	return f'./models/{architecture}/{architecture}_{target_dataset}_from_{source_dataset}.pth'

def load_feature_extractor(model_id:str) -> FeatureExtractor:
	fe = FeatureExtractor(model_id)
	return fe

def load_source_model(architecture:str, source_dataset:str) -> torch.nn.DataParallel:
	kwdargs = {}
	if architecture == 'googlenet':
		kwdargs = {'aux_logits': False, 'init_weights': False}

	if source_dataset in task_configuration.num_classes:
		num_classes = task_configuration.num_classes[source_dataset]
	else:
		num_classes = 1000

	# net = torch.nn.DataParallel(getattr(torchvision.models, architecture)(pretrained=False, num_classes=num_classes, **kwdargs))
	# net.load_state_dict(torch.load(get_source_model_path(architecture, source_dataset)))
	net = torch.nn.DataParallel()
	return net

def load_transfer_model(architecture:str, source_dataset:str, target_dataset:str) -> torch.nn.DataParallel:
	kwdargs = {}
	if architecture == 'googlenet':
		kwdargs = {'aux_logits': False, 'init_weights': False}

	net = torch.nn.DataParallel(getattr(torchvision.models, architecture)(pretrained=False, num_classes=task_configuration.num_classes[target_dataset], **kwdargs))
	net.load_state_dict(torch.load(get_transfer_model_path(architecture, source_dataset, target_dataset)))
	return net


def seed_all(seed:int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

def make_dirs(path: str):
	""" Why is this not how the standard library works? """
	path = os.path.split(path)[0]
	if path != "":
		if not os.path.exists(path):
			os.makedirs(path, exist_ok=True)




class CSVCache:
	def __init__(self, path:str, header:list, key:list, append:bool=True):
		self.path = path
		self.cache = {}
		self.key_fmt = [header.index(k) for k in key]
		self.header = header
		self.header_idx = {k: idx for idx, k in enumerate(header)}

		if not append and os.path.exists(path):
			os.remove(path)
		

		if os.path.exists(path):
			with open(path, 'r') as f:
				reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

				for idx, row in enumerate(reader):
					if idx > 0 and len(row) > 0:
						self.add_to_cache(row)

		else:
			make_dirs(path)
			self.write_row(header)

	def add_to_cache(self, row:list):
		key = tuple([row[i] for i in self.key_fmt])
		self.cache[key] = row

	def write_row(self, row:list):
		with open(self.path, 'a') as f:
			writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(row)
			self.add_to_cache(row)

	def exists(self, *args):
		return tuple([str(x) for x in args]) in self.cache

	def rewrite(self, path:str=None):
		""" Saves the current cache over the source file. """
		if path is None:
			path = self.path

		with open(path, 'w') as f:
			writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(self.header)

			for k, row in self.cache.items():
				writer.writerow(row)
