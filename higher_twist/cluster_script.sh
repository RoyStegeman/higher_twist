#!/bin/sh

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

for fit in ${fits[@]}; do
	$qsubpath -N "HT_"$fit script.sh $fit
done
