#!/bin/bash

#SBATCH --job-name=muvis

#SBATCH -e errors.txt
#SBATCH -o terminal_output.txt

python3 animate_image.py
