#!/bin/bash


nohup python run_exp.py \
	--model_tag codet5_base \
	--task concode \
	--sub_task none \
	> train.log 2>&1 &
