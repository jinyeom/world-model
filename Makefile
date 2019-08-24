DOCKERFILE=Dockerfile
IMAGE_NAME=world_model
CONTAINER_NAME=wm

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

run:
	docker run --runtime=nvidia -it --rm --name $(CONTAINER_NAME) -v $(shell pwd):/workspace $(IMAGE_NAME)