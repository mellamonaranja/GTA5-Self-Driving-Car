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
- Fix *object_detection* import issue
https://wikidocs.net/80978

```
wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O [YOUR_LOCAL_PATH]/google/protobuf/internal/builder.py
```
