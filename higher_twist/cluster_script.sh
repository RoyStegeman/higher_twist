#!/bin/sh

qsubpath=/exports/applications/gridengine/ge-8.6.5/bin/lx-amd64/qsub

fits=(
    "240530-01-fABMP22-CT-iterated",
    "240530-01-fABMP22R-CT-iterated",
    "240603-01-fABMP22-iterated",
    "240603-01-fABMP22R-iterated",
    "240611-01-fABMP22",
    "240611-01-fABMP22R"
)

for fit in ${fits[@]}; do
	$qsubpath -N "HT_"$fit script.sh $fit
done
