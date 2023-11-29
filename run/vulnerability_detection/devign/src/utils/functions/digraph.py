import subprocess


def create_digraph(name, nodes):
    with open('digraph.gv', 'w') as f:
        f.write("digraph {\n")
        digraph = ''

        for n_id, node in nodes.items():
            label = node.get_code() + "\n" + node.label
            f.write(f"\"{node.get_code()}\" [label=\"{label}\"]\n")
            for e_id, edge in node.edges.items():
                if edge.type != "Ast": continue

                if edge.node_in in nodes and edge.node_in != node.id:
                    n_in = nodes.get(edge.node_in)
                    label = "\"" + n_in.label + "\""
                    digraph += f"\"{node.get_code()}\" -> \"{n_in.get_code()}\" [label={label}]\n"
                '''
				if edge.node_out in nodes and edge.node_out != node.id:
					n_out = nodes.get(edge.node_out)
					label = "\"" + n_out.label + "\""
					digraph += f"\"{n_out.get_code()}\" -> \"{node.get_code()}\" [label={label}]\n"		
				'''
        f.write(digraph)
        f.write("}\n")
    subprocess.run(["dot", "-Tps", "digraph.gv", "-o", f"{name}.ps"], shell=False)


'''
def to_digraph(name, nodes):
    k_nodes = nodes.keys()
    code = {}
    connections = { "in" : dict.fromkeys(k_nodes), "out" : dict.fromkeys(k_nodes) }

	for n_id, node in nodes.items():
		#print(n_id, node.properties)
		connections = node.connections(connections, "Ast")
		code.update({n_id : node.get_code()})

	create_digraph(name, code, k_nodes, connections)
'''
