# VFX-HW1-HDR
### Introduction
First, we select a scene that has high dynamic range in order to create results which are clearly better than single exposure. Then, we proceed to align the photos and find the response function with different exposures. After computing our radiance map, we need to do tone mapping to show our results clearly.
### Source
* hw1_1.py: the main function to read all images and generate result, including the Paul's Debevec HDR recovery algorithm
* MTB.py: the Ward's MTB algorithm
* ToneMapping.py: the Durand and Dorsey’s Fast Bilateral Filtering algorithm for tone mapping
### How to run
* ```./run.sh [dir path]```
* dir path: the directory contain original images and list
* result will be placed under the same path
### Reference
[1] Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures, G. Ward, JGT 2003

[2] Fast Bilateral Filtering for the Display of High-Dynamic-Range Images
http://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/

[3] Recovering High Dynamic Range Radiance Maps from Photographs, Paul E. Debevec, Jitendra Malik, SIGGRAPH 1997
