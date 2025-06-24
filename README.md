## Installation

### Foundation Stereo
`requirements.txt`

### GraspNet
```bash
conda create -n cooscan python=3.11
conda activate cooscan
pip install robotic==0.2.9
```
Get the code.
```bash
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

### Contact-GraspNet
```bash
pip install pyrender==0.1.45
```

### Extras
might have to change these lines in `/home/denis/miniconda3/envs/cooscan/lib/python3.11/site-packages/transforms3d/quaternions.py`:
```bash
_MAX_FLOAT = np.maximum_sctype(np.float) -> _MAX_FLOAT = np.float64
_FLOAT_EPS = np.finfo(np.float).eps -> _FLOAT_EPS = np.finfo(float).eps
```

# Checkpoints
GraspNet (put into `./checkpoint/graspnet`)
- `checkpoint-rs.tar`
[Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)
FoundationStereo (put into `./checkpoints/foundationstereo`)
[Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing). Put the entire folder (e.g. `23-51-11`).
