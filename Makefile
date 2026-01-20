XGBOOST_SCIKIT_LEARN := us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn
XGBOOST_TAG := v2.1.2

IMAGE_ONNX := us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx
IMAGE_ONNX_TAG := v2.1.2

build_local_xgboost_scikit_learn:
	docker buildx build --tag $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG)_local -f docker-xgboost-scikit-learn/Dockerfile .

build_xgboost_scikit_learn:
	docker buildx build --platform linux/amd64 --tag $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG) -f docker-xgboost-scikit-learn/Dockerfile .

push_xgboost_scikit_learn: build_xgboost_scikit_learn
	docker push $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG)

build_local_image_onnx:
	docker buildx build --tag $(IMAGE_ONNX):$(IMAGE_ONNX_TAG)_local -f docker-image-onnx/Dockerfile .

build_image_onnx:
	docker buildx build --platform linux/amd64 --tag $(IMAGE_ONNX):$(IMAGE_ONNX_TAG) -f docker-image-onnx/Dockerfile .

push_image_onnx: build_image_onnx
	docker push $(IMAGE_ONNX):$(IMAGE_ONNX_TAG)
