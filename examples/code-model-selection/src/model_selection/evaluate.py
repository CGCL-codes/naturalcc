import traceback

import torchvision
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import pickle
import itertools
import time
import gc
import csv
from collections import defaultdict
import argparse
from typing import Dict
from prettytable import PrettyTable

from task_configuration import variables
# nlp_external
from selection_datasets import BalancedClassSampler, FixedBudgetSampler
import utils
from methods import TransferabilityMethod
from feature_extractor import FeatureExtractor

class ClassBalancedExperimentParams:
	""" Using a fixed number of instances per class. """

	def __init__(self, instances_per_class:int):
		self.instances_per_class = instances_per_class
		self.experiment_name = f'class_balanced_{self.instances_per_class}'

	def create_dataloader(self, dataset:str, batch_size:int, **kwdargs):
		return BalancedClassSampler(dataset, batch_size=batch_size, instances_per_class=self.instances_per_class, **kwdargs)
		

class FixedBudgetExperimentParams:
	""" Using a fixed budget probe size with classes distributed as evenly as possible. """

	def __init__(self, budget:int):
		self.budget = budget
		self.experiment_name = f'fixed_budget_{self.budget}'

	def create_dataloader(self, dataset:str, batch_size:int, **kwdargs):
		return FixedBudgetSampler(dataset, batch_size=batch_size, probe_size=self.budget, **kwdargs)


class Experiment:

	"""
	Runs the given method on each probe set and outputs score and timing information into ./results/.
	To evaluate the results, see metrics.py.

	Params:
	 - methods: A dictionary of methods to use.
	 - budget: The number of images in each probe set. Leave as default unless you want to extract your own probe sets.
	 - runs: The number of different probes sampled per transfer. Leave as default unless you want to extract your own probe sets.
	 - probe_only: If True, skips doing method computation and instead only extracts the probe sets.
	 - model_bank: Which model bank to use. Options are: "controlled" (default) and "all" (includes crowd-sourced).
	 - append: If false (default), the output file will be overwritten. Otherwise, it will resume from where it left off. When resuming, timing information will be lost.
	 - name: The name of the experiment. Defaults to the name of the probe set.
	"""

	def __init__(self, methods:Dict[str, TransferabilityMethod],
				 budget:int=500,
				 runs:int=5,
				 probe_only:bool=False,
				 model_bank:str='controlled',
				 append:bool=False,
				 name:str=None,
				 ):
		self.params = FixedBudgetExperimentParams(budget)
		self.runs = runs
		self.probe_only = probe_only
		self.model_bank = model_bank
		self.name = name if name is not None else self.params.experiment_name
		self.methods = methods
		self.dataloaders = {}
		# score_list: <model, <score, list> >
		self.score_list = {}

		key = ['Run', 'Model', 'Source Dataset', 'Target Dataset']
		headers = key + list(self.methods.keys())

		self.out_cache = utils.CSVCache(self.out_file, headers, key=key, append=append)

		self.times = defaultdict(list)
		
	def cache_path(self, model:str, source_dataset:str, target_dataset:str, run:int):
		return f'./cache/probes/{self.params.experiment_name}/{model}_{source_dataset}_{target_dataset}_{run}.pkl'

	@property
	def cur_cache_path(self):
		return self.cache_path(self.model_name, self.source_dataset, self.target_dataset, self.run)

	@property
	def out_file(self):
		return f'./results/{self.name}.csv'

	@property
	def timing_file(self):
		return f'./results/{self.name}_timing.pkl'

	def prep_nlp_fe(self):
		fe = utils.load_feature_extractor(self.model_name)
		if(fe.openai):
			return fe
		model = fe.model
		if(torch.cuda.is_available()):
			model.cuda()
		model.eval()

		def extract_feats(self, args):
			x = args[0]
			model._extracted_feats[x.get_device()] = x

		for name, module in model.named_modules():
			if isinstance(module, nn.Linear):
				module.register_forward_pre_hook(extract_feats)

		return fe





	def probe(self):
		""" Returns (and creates if necessary) probe data for the current run. """
		cache_path = self.cur_cache_path
		
		if os.path.exists(cache_path):
			with open(cache_path, 'rb') as f:
				return pickle.load(f)
		
		if self.model == None:
			self.fe = self.prep_nlp_fe()
			self.model = self.fe.model


		dataloader_key = (self.target_dataset, self.run)

		if dataloader_key not in self.dataloaders:
			utils.seed_all(2020 + self.run * 3037)
			dataloader = self.params.create_dataloader(self.target_dataset,
													   batch_size=8,
													   train=True,
													   pin_memory=True)

			self.dataloaders[dataloader_key] = dataloader
		dataloader = self.dataloaders[dataloader_key]
		if(self.fe.openai):
			all_y = []
			all_feats = []
			for x, y in tqdm(dataloader):
				try:
					preds = self.fe.openai_forward(x)
					all_y.append(y)
					# all_probs.append(torch.nn.functional.softmax(preds, dim=-1).cpu())
					all_feats.append(torch.tensor(preds))
				except Exception as e:
					print("Exception occurred")
					pass
			# raise NotImplementedError("OpenAI Not Implemented.")
		else:
			with torch.no_grad():
				all_y     = []
				all_feats = []
				# all_probs = []
				print("Probing...")
				for x, y in tqdm(dataloader):
					# Support for using multiple GPUs
					if(torch.cuda.is_available()):
						self.model._extracted_feats = [None] * torch.cuda.device_count()
					else:
						self.model._extracted_feats = [None]


					x = self.fe.Tokenize(x)
					input_ids = x['input_ids']
					attention_mask = x['attention_mask']
					if torch.cuda.is_available():
						input_ids = input_ids.cuda()
						attention_mask = attention_mask.cuda()
					preds = self.fe.forward(input_ids, attention_mask)
					all_y.append(y.cpu())
					# all_probs.append(torch.nn.functional.softmax(preds, dim=-1).cpu())
					all_feats.append(preds.cpu())
					# all_feats.append(torch.cat([x.cpu() for x in self.model._extracted_feats], dim=0))

		all_y     = torch.cat(all_y    , dim=0).numpy()
		all_feats = torch.cat(all_feats, dim=0).numpy()
		params = {
			'features': all_feats,
			'probs': {},
			'y': all_y,
			'source_dataset': self.source_dataset,
			'target_dataset': self.target_dataset,
			'model': self.model_name
		}
		utils.make_dirs(cache_path)
		with open(cache_path, 'wb') as f:
			pickle.dump(params, f)
		
		return params

	class Score:
		def __init__(self, model_name, model_score):
			self.model_name = model_name
			self.model_score = model_score

	def demo(self, model_name, method_list, score_list):
		if model_name not in self.score_list:
			self.score_list[model_name] = {}
		for i in range(len(method_list)):
			if method_list[i] not in self.score_list[model_name]:
				self.score_list[model_name][method_list[i]] = []
			self.score_list[model_name][method_list[i]].append(score_list[i])


	def ranking(self):
		x = PrettyTable()
		x.field_names = ["Recommendation", "Model ID", "Score"]
		score_list = []
		for model in self.score_list.keys():
			rank = self.score_list[model]
			score = np.mean(np.array(rank[list(rank.keys())[0]]))
			score_list.append(self.Score(model, score))
		score_list.sort(key=lambda x: x.model_score)
		score_list.reverse()
		print("Model Selection:")
		rank = 1
		for body in score_list:
			x.add_row([rank, body.model_name, body.model_score])
			rank = rank+1
		print(x)

	def evaluate(self):
		params = self.probe()

		if self.probe_only:
			return

		# if self.source_dataset == self.target_dataset:
		# 	return

		params['cache_path_fn'] = lambda model, source, target: self.cache_path(model, source, target, self.run)
		
		scores = [self.run, self.model_name, self.source_dataset, self.target_dataset]
		demo_method = []
		demo_inner = []
		for idx, (name, method) in enumerate(self.methods.items()):
			utils.seed_all(1010 + self.run * 2131)
			last_time = time.time()
			score = method(**params)
			scores.append(score)
			demo_method.append(name)
			demo_inner.append(score)
			self.times[name].append(time.time() - last_time)
		self.demo(self.model_name, demo_method, demo_inner)
		# print(f"SCORES: {scores}")
		self.out_cache.write_row(scores)

		
	def download_models(self):
		models = variables['Model']
		for i in models:
			print(i)
			fe = FeatureExtractor(i)

	def run(self):
		""" Run the methods on the data and then saves it to out_path. """
		last_model = None

		factors = [variables['Model'], variables['Source Dataset'], variables['Target Dataset'], list(range(self.runs))]

		iter_obj = []		

		# if self.model_bank == 'all':
		# 	for arch, source in nlp_external:
		# 		for target in variables['Target Dataset']:
		# 			for run in range(self.runs):
		# 				iter_obj.append((arch, source, target, run))

		iter_obj += list(itertools.product(*factors))

		for arch, source, target, run in tqdm(iter_obj):
			# RSA requires source-source extraction, so keep this out
			# if source == target:
			# 	continue
			try:
				print("Iteration:", arch, source, target, run)
				if self.out_cache.exists(run, arch, source, target):
					continue

				cur_model = (arch, source)
				if cur_model != last_model:
					self.model = None

				self.model_name = arch
				self.source_dataset = source
				self.target_dataset = target
				self.run = run

				self.evaluate()
			except Exception as e:
				print(f"-----# Error in {arch} #-----")
				traceback.print_exc()
			finally:
				gc.collect()

		self.ranking()
		for name, times in self.times.items():
			print(f'{name:20s}: {sum(times) / len(times): .3f}s average')
		
		with open(self.timing_file, 'wb') as f:
			pickle.dump(dict(self.times), f)
		

		

		


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--budget'    , help='Number of image in the probe set. Default is 500.', default=500, type=int)
	parser.add_argument('--runs'      , help='Number of probe sets sampled per transfer. Default is 5.', default=5, type=int)
	parser.add_argument('--probe_only', help='Set this flag if you only want to generate probe sets.', action='store_true')
	parser.add_argument('--model_bank', help='Which model bank to use. Options are "controlled" and "all". Default is "controlled".', default='controlled', type=str)
	args = parser.parse_args()

	Experiment(args.budget, runs=args.runs, probe_only=args.probe_only, model_bank=args.model_bank).run()


