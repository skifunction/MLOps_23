#!/bin/bash

IMAGE="digits:v"

#creating volume
docker volume create models

#building docker
docker build -t $IMAGE -f docker/dockerfile .
docker run -d --name MODELS -v models:/digits/models $IMAGE
docker cp MODELS:/digits/models/ /mnt/c/Users/sujay/ML_OPS/MLOps_23