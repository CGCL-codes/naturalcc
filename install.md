```shell script
# install conda
conda create -n ncc python=3.7
conda activate ncc

# download libraries:
# pytorch-cuda
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# pytorch-cpu
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# g++
sudo apt install g++
# cython
conda install -c anaconda cython

# others
pip install gdown ujson dpu_utils nltk psutil ruamel.yaml

```