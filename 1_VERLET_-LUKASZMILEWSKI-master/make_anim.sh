#!/bin/bash

ffmpeg -framerate 20 -i e_frames/f_%05d.png -c:v libx264 e_anim.mp4
ffmpeg -framerate 20 -i v_frames/f_%05d.png -c:v libx264 v_anim.mp4
ffmpeg -framerate 20 -i lf_frames/f_%05d.png -c:v libx264 lf_anim.mp4
