<p align="center">
  <br>
  <img src="docs/naturalcc_logo.png" width="400">
  <br>
</p>
<div align="center">
  <a href="https://xcodemind.github.io/">
    <img src="https://img.shields.io/badge/NaturalCC-0.6.0-green" alt="Version">
  </a>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.6-3776AB?logo=python" alt="Python">
  </a>
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.4-FF6F00?logo=pytorch" alt="pytorch">
  </a>
  <a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphGallery" alt="license">
  </a>

<a href="https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf">Paper</a>,
<a href="http://121.43.116.104:3000">Demo</a>,
<a href="https://xcodemind.github.io/team">About us-XCodeMind</a>

# NaturalCC - Natural Code Comprehension

</div>

## 📖 Vision

NaturalCC is a sequence modeling toolkit designed to bridge the gap between programming and natural languages through advanced machine learning techniques. It allows researchers and developers to train custom models for a variety of software engineering tasks, e.g., code generation, code completion, code summarization, code retrieval, code clone detection, and type inference.


## 🌟 Key Features:

- **Modular and Extensible Framework:** Built on the robust Fairseq's registry mechanism, allowing for easy adaptation and extension to diverse software engineering tasks.
- **Datasets and Preprocessing Tools+:** Offers access to a variety of clean, preprocessed benchmarks such as Human-Eval, CodeSearchNet, Python-Doc, and Py150. Comes equipped with scripts for feature extraction using compiler tools like LLVM.
- **Support for Large Code Models:** Incorporates state-of-the-art large code models like Code Llama, CodeT5, CodeGen, and StarCoder.
- **Benchmarking and Evaluation**: Benchmarks multiple downstream tasks (including code generation and code completion), with evaluation capabilities on well-known benchmarks using popular metrics like pass@k.
- **Optimized for Efficiency:** Employs the `NCCL` library and `torch.distributed` for high-efficiency model training across multiple GPUs. Supports both full-precision (`FP32`) and half-precision (`FP16`) computations to accelerate training and inference processes.
- **Enhanced Logging for Improved Debugging:** Advanced logging features to provide clear, detailed feedback during model training and operation, aiding in debugging and performance optimization.

## ✨ Latest News

- **[Nov 25, 2023]** **NaturalCC 2.0 Released!** Now compatible with [Transformers](https://github.com/huggingface/transformers) and supporting popular large code models like Code Llama, CodeT5, CodeGen, and StarCoder from [Hugging Face](https://huggingface.co). Access the previous version in the [ncc1]() branch.
- **[Apr 19, 2023]** Integrated the source code of "You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search" into NaturalCC.
- **[Jan 25, 2022]** Our paper introducing the NaturalCC toolkit was accepted at the ICSE 2022 Demo Track.
- **[May 10, 2022]** Merged the source code of "What Do They Capture? - A Structural Analysis of Pre-Trained Language Models for Source Code" into NaturalCC.

## 🛠️ Installation Guide

To get started with NaturalCC, ensure your system meets the following requirements:

- GCC/G++ version 5.0 or higher
- NVIDIA GPU, NCCL, and the Cuda Toolkit for training new models (optional but recommended)
- NVIDIA's apex library for faster training (optional)

Follow these steps to set up the environment.

1. (Optional) Creating conda environment
    ```shell
    conda create -n naturalcc python=3.6
    conda activate naturalcc
    ```

2. Building NaturalCC from source
    ```shell
    git clone https://github.com/CGCL-codes/naturalcc && cd naturalcc
    pip install -r requirements.txt
    pip install --editable ./
    ```
3. Installing Additional Dependencies
    ```shell
    pip install -q -U git+https://github.com/huggingface/transformers.git
    pip install -q -U git+https://github.com/huggingface/accelerate.git
    ```

4. HuggingFace Token for Certain Models

    For models like [StarCoder](https://github.com/bigcode-project/starcoder), a HuggingFace token is required. Log in to HuggingFace using:
    ```
    huggingface-cli login
    ```

## 🚀 Quick Start

### Example 1: Code Generation

1. Download the model checkpoint

    First, download the checkpoint of a specific large code model. For this example, we use [Codellama-7B](https://huggingface.co/camenduru/CodeLlama-7b).


2. Prepare the testing dataset

    Create a JSON file containing your test cases in the following format:

    ```json
    [
      {"input": "this is a"},
      {"input": "from tqdm import"},
      {"input": "def calculate("},
      {"input": "a = b**2"},
      {"input": "torch.randint"},
      {"input": "x = [1,2"}
    ]
    ```

3. Running the code generation scripts

    1. Initialize the task with the specific model and GPU device:

        ```python
        print('Initializing GenerationTask')
        task = GenerationTask(task_name="codellama_7b_code", device="cuda:0")
        ```

    2. Load the downloaded checkpoint into the task. Replace `ckpt_path` with the path to your downloaded checkpoint:

        ```python
        print('Loading model weights [{}]'.format(ckpt_path))
        task.from_pretrained(ckpt_path)
        ```

    3. Load your dataset. Replace `dataset_path` with the path to your dataset file:

        ```python
        print('Processing dataset [{}]'.format(dataset_path))
        task.load_dataset(dataset_path)
        ```

    4. Run the model and output the results. Replace `output_path` with your desired output file path:

        ```python
        task.run(output_path=output_path, batch_size=1, max_length=50)
        print('Output file: {}'.format(output_path))
        ```

### Example 2: Code Summarization

1. Download and process a dataset from ```datasets```, and follow the instructions from the README.md file.
    ```shell
    # ref: dataset/python_wan/README.md
    # download dataset
    bash dataset/python_wan/download.sh
    # clean data
    python -m dataset.python_wan.clean
    # cast data attributes into different files
    python -m dataset.python_wan.attributes_cast

    # ref: dataset/python_wan/summarization/README.md
    # save code tokens and docstirng tokens into MMAP format
    python -m dataset.python_wan.summarization.preprocess
    ```

2. Register your self-defined models
    - If you want to create a new model, please add your model at ```ncc/models``` and ```ncc/modules```.

    - If your training policy are more complex than we thought, you should update your criterions and training procedure at ```ncc/criterions``` and ```ncc/trainers```, respectively.
      <br>

      Do not forget to update your self defined module at ```ncc/XX/__init__.py```.

3. Training and inference.
    - Select a task and a model from [task list](run/) and follow the instructions in its README.md to start your learning.
    ```shell
    # ref: run/summarization/transformer/README.md
    # train
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.transformer.train -f config/python_wan/python > run/summarization/transformer/config/python_wan/python.log 2>&1 &
    # inference
    CUDA_VISIBLE_DEVICES=0 python -m run.summarization.transformer.eval -f config/python_wan/python -o run/summarization/transformer/config/python_wan/python.txt
    ```

> We also have more detailed [READMEs](examples) to start your tutorial of NaturalCC.

## 📚 Dataset

NaturalCC supports a diverse range of datasets, catering to various aspects of code analysis and processing. These datasets include:

- [Python (Wan et al.)](preprocessing/python_wan/README.md)
- [CodeSearchNet (Husain et al.)](preprocessing/codesearchnet/README.md)
- [CodeXGlue (Feng et al.)](preprocessing/codexglue/code_to_text/README.md)
- [Py150 (official processed)](preprocessing/py150/README.md) [(raw)](preprocessing/raw_py150/README.md)
- [OpenCL (Grewe et al.)](preprocessing/opencl/README.md)
- [Java (Hu et, al.)](preprocessing/java_hu/README.md)
- [Stack Overflow](preprocessing/stackoverflow/README.md)
- [DeepCS (Gu et al.)](preprocessing/deepcs)
- [AVATAR (Ahmad et al.)](preprocessing/avatar)
- [StackOverflow (Iyer et al.)](preprocessing/stackoverflow)


## 🤝 Contributor

We warmly welcome contributions to NaturalCC! Your involvement is essential for keeping NaturalCC innovative and accessible. 

We're grateful to all our amazing contributors who have made this project what it is today!

<a href="https://github.com/CGCL-codes/naturalcc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CGCL-codes/naturalcc&r="  width="800px"/>
</a>

## 💡 FAQ

If you have any questions or encounter issues, please feel free to reach out. For quick queries, you can also check our `Issues` page for common questions and solutions.

## 😘 License and Acknowledgement

**License:** NaturalCC is open-sourced under the [MIT-licensed](https://github.com/CGCL-codes/naturalcc/blob/master/LICENSE.txt). This permissive license applies not only to the toolkit itself but also to the pre-trained models provided within.

**Acknowledgements:** We extend our heartfelt gratitude to the broader open-source community, particularly drawing inspiration from projects like [Fairseq](https://github.com/pytorch/fairseq) for their advanced sequence-to-sequence models, and [AllenNLP](https://allennlp.org) for their robust NLP components. Their groundbreaking work has been instrumental in shaping the development of NaturalCC.

## 📄 Citation

We're thrilled that you're interested in using NaturalCC for your research or applications! Citing our work helps us to grow and continue improving this toolkit. You can find more in-depth details about NaturalCC in our [paper](https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf).

If you use NaturalCC in your research, please consider citing our paper. Below is the BibTex format for citation:

```
@inproceedings{wan2022naturalcc,
  title={NaturalCC: An Open-Source Toolkit for Code Intelligence},
  author={Yao Wan and Yang He and Zhangqian Bi and Jianguo Zhang and Yulei Sui and Hongyu Zhang and Kazuma Hashimoto and Hai Jin and Guandong Xu and Caiming Xiong and Philip S. Yu},
  booktitle={Proceedings of 44th International Conference on Software Engineering, Companion Volume},
  publisher=ACM,
  year={2022}
}
```
