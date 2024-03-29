#!/bin/sh
#$ -cwd                  
#$ -l h_rt=10:00:00 
#$ -l h_vmem=10G 
#$ -o /exports/csce/eddie/ph/groups/nnpdf/Users/ac/codes/alphas_covmat/higher_twist/logs
#$ -e /exports/csce/eddie/ph/groups/nnpdf/Users/ac/codes/alphas_covmat/higher_twist/logs
#$ -m eas
#$ -M amedeochiefa@gmail.com

source /exports/csce/eddie/ph/groups/nnpdf/Users/ac/miniconda3/bin/activate nnpdf
CURRENTDIR=/exports/csce/eddie/ph/groups/nnpdf/Users/ac/codes/alphas_covmat/higher_twist

python posteriors.py $1 $CURRENTDIR

