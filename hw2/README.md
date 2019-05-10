## Project #2: Image Stitching
### Intro
In this project, we will implement image stitching to form panoramic image by combining the overlapping regions of multiple photos.

### Run
- input: ```grass/``` or ```dept/```
- execute: ```python3 main.py [input]```
- output: ```result/```

### Source
- main.py: read file and run through each step
- c_warp.py: do cylindrical warping and final blending
- detection.py: run Harris Corner detection and generate feature
- local_similarity.py: do feature matching with kd-tree
