#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
mkdir replications_vae_martina_Dekel
cd replications_vae_martina_Dekel

for i in {1..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i
    ln -s ../../matlab ./

    python ../../vanilla_vae.py 0
    cd ../
done

