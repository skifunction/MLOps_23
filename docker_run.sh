#!/bin/bash

IMAGE="digits:v"

docker volume create models

docker build -t $IMAGE -f docker/dockerfile .
docker run -d --name MODELS -v models:/digits/models $IMAGE
docker cp MODELS:/digits/models/ /mnt/c/Users/sujay/ML_OPS/MLOps_23