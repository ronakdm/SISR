#!/bin/bash

exp_name=multitarget_simulated_cond_5_incr
for job_id in 0 1; 
do
    python experiments/run_array.py $job_id $exp_name
done