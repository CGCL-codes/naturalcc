class Properties:
	def __init__(self, props, indentation):
		self.size = len(props)
		self.indentation = indentation + 1
		self.pairs = {prop["key"]: prop["value"] for prop in props}
	
	def __str__(self):
		indentation = self.indentation * "\t"
		string = ""

		for prop in self.pairs:
			string += f"\n{indentation}Property - {prop} : {self.pairs[prop]}"

		return f"{indentation}{string}\n"

	def code(self):
		if self.has_code():
			code = self.pairs["CODE"]
			if self.has_type() and self.get_type() != "ANY" and self.get_type() not in code:
				code = f"{self.get_type()} {code}"
			return code
		return None

	def get_type(self):
		return self.pairs.get("TYPE_FULL_NAME")

	def has_type(self):
		return "TYPE_FULL_NAME" in self.pairs

	def has_code(self):
		return "CODE" in self.pairs

	def line_number(self):
		return self.pairs["LINE_NUMBER"] if self.has_line_number() else None

	def has_line_number(self):
		return "LINE_NUMBER" in self.pairs

	def column_number(self):
		return self.pairs["COLUMN_NUMBER"] if self.has_column_number() else None

	def has_column_number(self):
		return "COLUMN_NUMBER" in self.pairs

	def get(self):
		return self.pairs

	def get_operator(self):
		value = self.pairs.get("METHOD_FULL_NAME")
		if value is None:
			return value
		if ("<operator>" in value) or ("<operators>" in value):
			return value.split(".")[-1] 
		return None
