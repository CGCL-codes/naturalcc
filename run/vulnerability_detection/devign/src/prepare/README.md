# Description

This module contains scripts used for transforming the raw code from the dataset into Code Property Graphs.

## cpg_generator.py
This script launches and queries the Joern server that deals with the conversion of the code.
It takes as input a Pandas Dataframe containing code functions which are written to a folder.
The Joern is queried to create the CPG from the functions in the folder and returns the result into json format.

## The JSON generated has the following keys:
"functions": Array of all methods contained in the currently loaded CPG
 - |_ "function": Method name as String
 - |_ "id": Method id as String (String representation of the underlying Method node)
 - |_ "AST": Array of all nodes connected via AST edges
 - |_ "CFG": Array of all nodes connected via CFG edges
 - |_ "PDG": Array of all nodes that are reachable by data-flow or control-flow from the current method locals
  
 - Nodes have the following keys:
     - |_ "id": Node id as String (String representation of the underlying AST node)
     - |_ "properties": Array of properties of the current node as key-value pair
     - |_ "edges": Array of all AST edges where the current node is referenced as inVertex or outVertex
         - |_ "id": Edge id as String (String representation of the AST edge)
         - |_ "in": Node id as String of the inVertex node (String representation of the inVertex node)
         - |_ "out": Node id as String of the outVertex node (String representation of the outVertex node)
