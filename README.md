# Lane and Obstacle detection on GTA V

## Find the Vanishing point
<img width="402" alt="image" src="https://github.com/mellamonaranja/GTA5-Self-Driving-Car/assets/55770526/d0374c5d-456a-4407-9f12-3e105526c01e">

## Lane detection for active assistance during driving.

![Lane_detect2-ezgif com-video-to-gif-converter](https://github.com/mellamonaranja/GTA5-Self-Driving-Car/assets/55770526/d6035de5-fbc0-4d7c-8dd1-a157d44132bb)


## Lane and obstacle detection for active assistance during driving.

![Lane_object_detect-ezgif com-video-to-gif-converter](https://github.com/mellamonaranja/GTA5-Self-Driving-Car/assets/55770526/7257ee97-7aaa-4630-97d2-434a8e59e0c6)

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

### USAGE EXAMPLE

Accompanying article https://towardsdatascience.com/copilot-driving-assistance-635e1a50f14
