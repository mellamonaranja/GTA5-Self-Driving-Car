# GTAVAI

## requirements

- python 3.9.18
- conda

## development

1. Create conda environment

```
conda create -n "GTA" python=3.9.18 ipython
```

2. Install packages

```
pip install -r requirements.txt
```

### Notes

- `requirements.txt` includes the required dependencies
- `requirements_f.txt` includes the freeze dependencies after installing
- If you receive the warning **dlerror: cudart64_110.dll not found** (https://github.com/tensorflow/tensorflow/issues/57103), execute the following command
```
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.0
```
