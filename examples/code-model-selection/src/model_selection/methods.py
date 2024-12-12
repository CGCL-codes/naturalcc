import numpy as np
import pickle
import pandas as pd

import task_configuration

import scipy.stats
import sklearn.neighbors
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
import torch
from tqdm import tqdm
from torch import nn

# This is for Logistic so it doesn't complain that it didn't converge
import warnings

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


def split_data(data: np.ndarray, percent_train: float):
	split = data.shape[0] - int(percent_train * data.shape[0])
	return data[:split], data[split:]


class TransferabilityMethod:
	def __call__(self,
				 features: np.ndarray, probs: np.ndarray, y: np.ndarray,
				 source_dataset: str, target_dataset: str, model: str,
				 cache_path_fn) -> float:
		self.features = features
		self.probs = probs
		self.y = y

		self.source_dataset = source_dataset
		self.target_dataset = target_dataset
		self.model_name = model

		self.cache_path_fn = cache_path_fn

		# self.features = sklearn.preprocessing.StandardScaler().fit_transform(self.features)

		return self.forward()

	def forward(self) -> float:
		raise NotImplementedError


def feature_reduce(features: np.ndarray, f: int = None) -> np.ndarray:
	"""
	Use PCA to reduce the dimensionality of the features.

	If f is none, return the original features.
	If f < features.shape[0], default f to be the shape.
	"""
	if f is None:
		return features

	if f > features.shape[0]:
		f = features.shape[0]

	return sklearn.decomposition.PCA(
		n_components=f,
		svd_solver='randomized',
		random_state=1919,
		iterated_power=1).fit_transform(features)


class LEEP(TransferabilityMethod):
	"""
	LEEP: https://arxiv.org/abs/2002.12462

	src ('probs', 'features') denotes what to use for leep.

	normalization ('l1', 'softmax'). The normalization strategy to get everything to sum to 1.
	"""

	def __init__(self, n_dims: int = None, src: str = 'probs', normalization: str = None, use_sigmoid: bool = False):
		self.n_dims = n_dims
		self.src = src
		self.normalization = normalization
		self.use_sigmoid = use_sigmoid

	def forward(self) -> float:
		theta = getattr(self, self.src)
		y = self.y

		n = theta.shape[0]
		n_y = constants.num_classes[self.target_dataset]

		# n             : Number of target data images
		# n_z           : Number of source classes
		# n_y           : Number of target classes
		# theta [n, n_z]: The source task probabilities on the target images
		# y     [n]     : The target dataset label indices {0, ..., n_y-1} for each target image

		unnorm_prob_joint = np.eye(n_y)[y, :].T @ theta  # P(y, z): [n_y, n_z]
		unnorm_prob_marginal = theta.sum(axis=0)  # P(z)   : [n_z]
		prob_conditional = unnorm_prob_joint / unnorm_prob_marginal[None, :]  # P(y|z) : [n_y, n_z]

		leep = np.log((prob_conditional[y] * theta).sum(axis=-1)).sum() / n  # Eq. 2

		return leep


class NegativeCrossEntropy(TransferabilityMethod):
	""" NCE: https://arxiv.org/pdf/1908.08142.pdf """

	def forward(self, eps=1e-5) -> float:
		z = self.probs.argmax(axis=-1)

		n = self.y.shape[0]
		n_y = task_configuration.num_classes[self.target_dataset]
		n_z = task_configuration.num_classes[self.source_dataset]

		prob_joint = (np.eye(n_y)[self.y, :].T @ np.eye(n_z)[z, :]) / n + eps
		prob_marginal = np.eye(n_z)[z, :].sum(axis=0) / n + eps

		NCE = (prob_joint * np.log(prob_joint / prob_marginal[None, :])).sum()

		return NCE


class HScore(TransferabilityMethod):
	""" HScore from https://ieeexplore.ieee.org/document/8803726 """

	def __init__(self, n_dims: int = None, use_published_implementation: bool = False):
		self.use_published_implementation = use_published_implementation
		self.n_dims = n_dims

	def getCov(self, X):
		X_mean = X - np.mean(X, axis=0, keepdims=True)
		cov = np.divide(np.dot(X_mean.T, X_mean), len(X) - 1)
		return cov

	def getHscore(self, f, Z):
		Covf = self.getCov(f)
		g = np.zeros_like(f)
		for z in range(task_configuration.num_classes[self.target_dataset]):
			idx = (Z == z)
			if idx.any():
				Ef_z = np.mean(f[idx, :], axis=0)
				g[idx] = Ef_z

		Covg = self.getCov(g)
		score = np.trace(np.dot(np.linalg.pinv(Covf, rcond=1e-15), Covg))

		return score

	def get_hscore_fast(self, eps=1e-8):
		# The original implementation of HScore isn't properly vectorized, so do that here
		cov_f = self.getCov(self.features)
		n_y = task_configuration.num_classes[self.target_dataset]

		# Vectorize the inner loop over each class
		one_hot_class = np.eye(n_y)[self.y, :]  # [#probe, #classes]
		class_counts = one_hot_class.sum(axis=0)  # [#classes]

		# Compute the mean feature per class
		mean_features = (one_hot_class.T @ self.features) / (class_counts[:, None] + eps)  # [#classes, #features]

		# Redistribute that into the original features' locations
		g = one_hot_class @ mean_features  # [#probe, #features]
		cov_g = self.getCov(g)

		score = np.trace(np.linalg.pinv(cov_f, rcond=1e-15) @ cov_g)

		return score

	def forward(self):
		self.features = feature_reduce(self.features, self.n_dims)

		scaler = sklearn.preprocessing.StandardScaler()
		self.features = scaler.fit_transform(self.features)

		if self.use_published_implementation:
			return self.getHscore(self.features, self.y)
		else:
			return self.get_hscore_fast()


class kNN(TransferabilityMethod):
	"""
	k Nearest Neighbors with hold-one-out cross-validation.

	Metric can be one of (euclidean, cosine, cityblock)

	This method supports VOC2007.
	"""

	def __init__(self, k: int = 1, metric: str = 'l2', n_dims: int = None):
		self.k = k
		self.metric = metric
		self.n_dims = n_dims

	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)

		dist = sklearn.metrics.pairwise_distances(self.features, metric=self.metric)
		idx = np.argsort(dist, axis=-1)

		# After sorting, the first index will always be the same element (distance = 0), so choose the k after
		idx = idx[:, 1:self.k + 1]

		votes = self.y[idx]
		preds, counts = scipy.stats.mode(votes, axis=1)

		n_data = self.features.shape[0]

		preds = preds.reshape(n_data, -1)
		counts = counts.reshape(n_data, -1)
		votes = votes.reshape(n_data, -1)

		preds = np.where(counts == 1, votes, preds)

		return 100 * (preds == self.y.reshape(n_data, -1)).mean()
# return -np.abs(preds - self.y).sum(axis=-1).mean() # For object detection


class SplitkNN(TransferabilityMethod):
	""" k Nearest Neighbors using a train-val split using sklearn. Only supports l2 distance. """

	def __init__(self, percent_train: float = 0.5, k: int = 1, n_dims: int = None):
		self.percent_train = percent_train
		self.k = k
		self.n_dims = n_dims

	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)

		train_x, test_x = split_data(self.features, self.percent_train)
		train_y, test_y = split_data(self.y, self.percent_train)

		nn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.k).fit(train_x, train_y)
		return 100 * (nn.predict(test_x) == test_y).mean()


class Logistic(TransferabilityMethod):
	""" Logistic classifier using a train-val split using sklearn. """

	def __init__(self, percent_train: float = 0.5, n_dims: int = None):
		self.percent_train = percent_train
		self.n_dims = n_dims

	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)

		train_x, test_x = split_data(self.features, self.percent_train)
		train_y, test_y = split_data(self.y, self.percent_train)

		logistic = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs',
														   max_iter=20, tol=1e-1).fit(train_x, train_y)
		return 100 * (logistic.predict(test_x) == test_y).mean()


##################################
device='cuda'
class InputFeature():
	def __init__(self, input_ids, label):
		self.input_ids = input_ids
		self.label = label

class ClassifyDataset(torch.utils.data.Dataset):
	def __init__(self, data):
		self.examples = [] # data
		for i in tqdm(data):
			if(torch.cuda.is_available()):
				self.examples.append(InputFeature(torch.tensor(i[0],device=device),torch.tensor(i[1],device=device)))
	def __len__(self):
		return len(self.examples)
	def __getitem__(self, i):
		return (self.examples[i].input_ids,
				self.examples[i].label)

class LinearNetwork(nn.Module):
	def __init__(self, input_dim, n_layers, hidden_dim, output_dim):
		super(LinearNetwork, self).__init__()
		self.n_layers = n_layers
		if (n_layers == 1):
			self.fe = nn.Linear(input_dim, output_dim)
			self.hidden = []
			self.be = None
		else:
			self.fe = nn.Linear(input_dim, hidden_dim)
			self.be = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))
			self.hidden = []
			for i in range(n_layers - 2):
				self.hidden.append(nn.ReLU())
				self.hidden.append(nn.Linear(hidden_dim, hidden_dim))

	def forward(self, x):
		if (self.n_layers == 1):
			return self.fe(x)
		else:
			outputs = self.fe(x)
			for hid in self.hidden:
				outputs = hid(outputs)
			return self.be(outputs)


def linear_evaluate(model, eval_dataset):
	eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=10)
	model.eval()
	logits = []
	labels = []
	eval_loss = 0.0
	nb_eval_steps = 0
	for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
		with torch.no_grad():
			outputs = model(batch[0].to(device))
			# print("Outputs:", outputs)
			loss = torch.nn.CrossEntropyLoss()(outputs, batch[1].to(device))
			eval_loss += loss.mean().item()
			logits.append(outputs.cpu().numpy())
			labels.append(batch[1].cpu().numpy())
		nb_eval_steps += 1
	logits = np.concatenate(logits, 0)
	labels = np.concatenate(labels, 0)
	preds = np.argmax(logits, axis=1)
	# preds = logits[:, 1] > 0.5
	eval_acc = np.mean(labels == preds)
	eval_loss = eval_loss / nb_eval_steps
	perplexity = torch.tensor(eval_loss)

	result = {
		"eval_loss": float(perplexity),
		"eval_acc": round(eval_acc, 4)}
	return result


def linear_classify(train_X,
					train_Y,
					eval_X,
					eval_Y,
					n_layers=1,
					hidden_dim=512,
					epoches=10,
					lr=5e-4):
	train_input = []
	eval_input = []
	# test_input = []
	X = train_X
	y = train_Y

	for i in range(len(train_X)):
		train_input.append((train_X[i],train_Y[i]))

	for i in range(len(eval_X)):
		eval_input.append((eval_X[i], eval_Y[i]))

	train_dataset = ClassifyDataset(train_input)
	eval_dataset = ClassifyDataset(eval_input)
	# test_dataset = ClassifyDataset(test_input)
	input_dim = len(X[0])
	num_classes = len(set(y))
	model = LinearNetwork(input_dim, n_layers, hidden_dim, num_classes)
	model.to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	for epoch in range(epoches):
		dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
		bar = tqdm(dataloader, total=len(dataloader), desc="Training")
		model.train()
		for batch in bar:
			optimizer.zero_grad()
			outputs = model(batch[0].to(device))
			loss = torch.nn.CrossEntropyLoss()(outputs, batch[1].to(device))
			loss.backward()
			optimizer.step()
			bar.set_description(f"[{epoch}] Train loss {round(loss.item(), 3)}")
		# evaluation
		print(linear_evaluate(model, eval_dataset))
	# test
	model.eval()
	acc = linear_evaluate(model, eval_dataset)['eval_acc']
	return acc


class Linear(TransferabilityMethod):
	""" Logistic classifier using a train-val split using sklearn. """

	def __init__(self, percent_train: float = 0.5, n_dims: int = None,
				 hidden_dim=2048,
				 epoches=100,
				 lr=5e-4
				 ):
		self.percent_train = percent_train
		self.n_dims = n_dims
		self.hidden_dim = hidden_dim
		self.epoches = epoches
		self.lr = lr

	def forward(self) -> float:
		self.features = feature_reduce(self.features, self.n_dims)

		train_x, test_x = split_data(self.features, self.percent_train)
		train_y, test_y = split_data(self.y, self.percent_train)
		acc = linear_classify(train_x,
							  train_y,
							  test_x,
							  test_y,
							  hidden_dim=self.hidden_dim,
							  epoches = self.epoches,
							  lr = self.lr
							  )
		return acc


class PARC(TransferabilityMethod):
	"""
	Computes PARC, a variation of RSA that uses target labels instead of target features to cut down on training time.
	This was presented in this paper.

	This method supports VOC2007.
	"""

	def __init__(self, n_dims: int = None, fmt: str = ''):
		self.n_dims = n_dims
		self.fmt = fmt

	def forward(self):
		self.features = feature_reduce(self.features, self.n_dims)

		num_classes = task_configuration.num_classes[self.target_dataset]
		labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

		return self.get_parc_correlation(self.features, labels)

	def get_parc_correlation(self, feats1, labels2):
		scaler = sklearn.preprocessing.StandardScaler()

		feats1 = scaler.fit_transform(feats1)

		rdm1 = 1 - np.corrcoef(feats1)
		rdm2 = 1 - np.corrcoef(labels2)

		lt_rdm1 = self.get_lowertri(rdm1)
		lt_rdm2 = self.get_lowertri(rdm2)

		return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

	def get_lowertri(self, rdm):
		num_conditions = rdm.shape[0]
		return rdm[np.triu_indices(num_conditions, 1)]

