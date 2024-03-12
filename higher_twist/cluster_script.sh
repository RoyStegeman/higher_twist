#!/bin/sh
#$ -N HT_posterior
#$ -cwd
#$ -l h_rt=10:00:00
#$ -l h_vmem=10G
#$ -o /exports/csce/eddie/ph/groups/nnpdf/Users/ac/comparefits/logs
#$ -e /exports/csce/eddie/ph/groups/nnpdf/Users/ac/comparefits/logs
#$ -m eas
#$ -M amedeochiefa@gmail.com

source /exports/csce/eddie/ph/groups/nnpdf/Users/ac/miniconda3/bin/activate nnpdf
script=/Users/s2569857/Codes/NNPDF4.0/alphas_covmat/higher_twist/posteriors.py
qsubpath=/exports/applications/gridengine/ge-8.6.5/bin/lx-amd64/qsub

fits=(
    "240308-01-ach-ht-5pt-nom"
    "240308-02-ach-ht-5pt-low"
    "240308-03-ach-ht-5pt-high"
    "240308-04-ach-ht-5pt-lowQ2"
    "240308-05-ach-ht-5pt-lowW2"
    "240308-06-ach-ht-5pt-nom-dis"
    "240308-07-ach-ht-5pt-low-dis"
    "240308-08-ach-ht-5pt-high-dis"
    "240308-09-ach-ht-5pt-lowQ2-dis"
    "240308-10-ach-ht-5pt-lowW2-dis"
)

$qsubpath -t 1-10 -l h_vmem=2500M -pe sharedmem 8 python posteriors.py ${fits[$SGE_TASK_ID]} ./