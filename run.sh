sudo docker build -t datamining .
sudo docker run -v `pwd`/:/app -p 80:80 datamining