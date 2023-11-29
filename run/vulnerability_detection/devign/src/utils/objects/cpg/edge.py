class Edge:
	def __init__(self, edge, indentation):
		self.id = edge["id"].split(".")[-1]
		self.type = self.id.split("@")[0]
		self.node_in = edge["in"].split(".")[-1]
		self.node_out = edge["out"].split(".")[-1]
		self.indentation = indentation + 1

	def __str__(self):
		indentation = self.indentation * "\t"
		return f"\n{indentation}Edge id: {self.id}\n{indentation}Node in: {self.node_in}\n{indentation}Node out: {self.node_out}\n"