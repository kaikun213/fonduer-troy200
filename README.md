# Knowledge Base augmentation from Spreadsheets 
## Fonduer on TROY200 spreadsheet data set

This project uses [Fonduer](https://github.com/HazyResearch/fonduer) to extract information from spreadsheets.
It aims to extract the data values and corresponding row/column headers from the [TROY200](https://data.mendeley.com/datasets/ydcr7mcrtp/6) data set.

## Installation

Clone the repository and execute inside the folder.
```
docker build -t kaikun/fonduer-troy200 .
docker-compose up
```
It will spin up a docker container with the jupyter notebook and one running the postgres database.
The jupyter notebook access token is printed in the console.
The credentials for the postgres database are username `user` and password `venron` by default.
They can be changed in the `docker-compose.yml` and will need to be adjusted in the notebook too.

If `docker-compose` is not installed, run the two containers independently within a local network to be able to access the service by container name. E.g.:

1.) Create a user-defined bridge network
```
docker network create my-net
```

2.) Run postgres (latest) docker image
```
docker pull postgres
mkdir -p $HOME/docker/volumes/pg
docker run --rm --name postgres --network my-net -e POSTGRES_PASSWORD=venron -e POSTGRES_USER=user -d -p 5432:5432 -v $HOME/docker/volumes/pg:/var/lib/postgresql/data postgres
```

3.) Run the code
```
docker build -t kaikun/fonduer-troy200 .
docker run --rm --name app-docker --network my-net -e PGPASSWORD=venron -e PGUSER=user -d -p 8890:8888 kaikun/fonduer-troy200
```

4.) Check the access token if no password for the notebook server is set
```
docker ps
docker logs CONTAINER_ID
```

Now the jupyter notebook will be accessible on `localhost:8890` and able to connect directly to the database. 
Please note that it is discouraged to use the database password as environment variable as it is potentially exposed to other users on the network.
Especially if you use shared resources switch to a [password file](https://www.postgresql.org/docs/current/libpq-pgpass.html). 

## Run on Server

In case the code runs on a remote server and you need to connect via SSH, simply use SSH tunnelling to connect to the notebook server.

```
ssh -N -L 8080:localhost:8890 USER@HOST_IP -p PORT
```

This will make the notebook server accessible on the the `8080` port on your local machine. Simply access in the browser `localhost:8080`.
