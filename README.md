## Welcome!
This repository contains source materials for Teleoperation System* (TS; name omitted for submission), an open-source teleoperation system with force feedback.

Feel free to visit our [website](https://hidden-submission.github.io/Teleoperation-System/) for more information!

### Repository content

Code for teleoperation and dataset collection is available for reviewers. 3D models, electronic components, and ready-to-use microcontroller firmware omitted for submission.

### Hardware & Assembly Tutorial

The tutorial is omitted for submission, but you can explore it after the paper is published! We offer a complete materials list and a clear video guide to assemble TS from the ground up.

### Installation Guide

Install system libraries:

```bash
sudo apt update
sudo apt install ffmpeg libx264-dev v4l-utils
```

Clone the repo:
```bash
git clone https://github.com/hidden-submission/Teleoperation-System.git
cd Teleoperation-System
```
Create a new conda environment and install the dependencies. We recommend Python 3.10:

```bash
conda create -n teleoperation_system python=3.10 -y
conda activate teleoperation_system
pip install -e .
pip install datasets --no-deps
```

### Citation

```
Omitted for submission
```