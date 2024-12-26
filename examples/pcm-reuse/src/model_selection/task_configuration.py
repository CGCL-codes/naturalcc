import methods

n_runs = 2
budget = 5000

variables = {
	'Source Dataset' : [
		'codexglue_defect_detection'
	],
	'Target Dataset': [
	],

	'Model': [
		'uclanlp/plbart-csnet',
		'uclanlp/plbart-large',
		'uclanlp/plbart-base',
		'microsoft/codebert-base',
		'Salesforce/codet5-base',
		'Salesforce/codet5-small',
		'Salesforce/codet5-large',
	],
}

method_map = {
	'Logistic': methods.Logistic(),
	'1-NN': methods.kNN(k=1),
	'3-NN': methods.kNN(k=3),
	'5-NN': methods.kNN(k=5),
	'HScore': methods.HScore(),
	'PARC': methods.PARC()
}

selection_methods = {}

num_classes = {
	'codexglue_defect_detection': 2
}
