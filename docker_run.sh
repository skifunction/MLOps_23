sudo docker build -t digits:v1 -f docker/dockerfile .

sudo docker run -v /mnt/c/Users/sujay/ML_OPS/MLOps_23/models/:/digits/models digits:v1

docker stop digits:v1