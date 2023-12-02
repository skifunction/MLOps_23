sudo docker build -t dependencyimage:latest -f docker/dockerfile .
sudo docker build -t finaldockerfile:latest -f docker/dockerfile.final .

sudo docker run -v /mnt/c/Users/sujay/ML_OPS/MLOps_23/models/:/digits/models dependencyimage:latest
sudo docker run -v /mnt/c/Users/sujay/ML_OPS/MLOps_23/models/:/digits/models finaldockerfile:latest