# ProGraML: Program Graphs for Machine Learning

## Notes
This project is from https://github.com/ChrisCummins/ProGraML, which is a powerful 
tool to generate graph from program, as well as includes some GNN baselines. 
One troublesome thing is that this tool is 
built on Bazel, which is unfriendly to some developers who are unfamiliar with it.

To this end, I separate this tool into two independent parts: one for graph generation 
(implemented in C and compiled using Bazel), another for baselines implementation
 (implemented in PyTorch).

The part for graph generation is put in the `third_party` folder of `naturalcc`.
The part for baselines is integrated in the `models` and `tasks` of `naturalcc`.

## Development IDE
Since this project is implemented in C and built using Bazel, we recommend to develop
it using VSCode rather than PyCharm. This is because the VSCode has a better support for Bazel. 

## Building
- Step 0: Generate proto python files based on the defined proto files (in `third_party/programl/programl/proto`)

```
cd naturalcc-dev
python setup.py generate_proto
```

- Step 1: Build the programl via Bazel
```
cd naturalcc-dev/third_party/programl
bazel build bin/main
bazel build //... (currently exist bugs)
```

## Usage


#### End-to-end C++ flow

In the manner of Unix Zen, creating and manipulating ProGraML graphs is done
using [command-line tools](/Documentation/bin) which act as filters,
reading in graphs from stdin and emitting graphs to stdout. The structure for
graphs is described through a series of [protocol
buffers](/Documentation/ProtocolBuffers.md).

This section provides an example step-by-step guide for generating a program
graph for a C++ application.

1. Install LLVM-10 and the ProGraML [command line tools](#command-line-tools).
2. Compile your C++ code to LLVM-IR. The way to do this to modify your build
   system so that clang is passed the `-emit-llvm -S` flags. For a
   single-source application, the command line invocation would be:
```
$ clang-10 -emit-llvm -S -c my_app.cpp -o my_app.ll
```
   For a multi-source application, you can compile each file to LLVM-IR
   separately and then link the results. For example:
```
$ clang-10 -emit-llvm -S -c foo.cpp -o foo.ll
$ clang-10 -emit-llvm -S -c bar.cpp -o bar.ll
$ llvm-link foo.ll bar.ll -S -o my_app.ll
```
3. Generate a ProGraML graph protocol buffer from the LLVM-IR using the
   [llvm2graph](https://github.com/ChrisCummins/ProGraML/blob/development/Documentation/bin/llvm2graph.txt)
   commnand:
```
$ llvm2graph < my_app.ll > my_app.pbtxt
```
   The generated file `my_app.pbtxt` uses a human-readable
   [ProgramGraph](https://github.com/ChrisCummins/ProGraML/blob/development/programl/proto/program_graph.proto)
   format which you can inspect using a text editor. In this case, we will
   render it to an image file using Graphviz.

4. Generate a Graphviz dotfile from the ProGraML graph using
   [graph2dot](https://github.com/ChrisCummins/ProGraML/blob/development/Documentation/bin/graph2dot.txt):
```
$ graph2dot < my_app.pbtxt > my_app.dot
```
5. Render the dotfile to a PNG image using Graphviz:
```
$ dot -Tpng my_app.dot -o my_app.png
```