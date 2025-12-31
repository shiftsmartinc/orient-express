build_local_xgboost_scikit_learn:
	docker buildx build --tag us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:latest -f docker-xgboost-scikit-learn/Dockerfile .

build_xgboost_scikit_learn:
	docker buildx build --platform linux/amd64 --tag us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:latest -f docker-xgboost-scikit-learn/Dockerfile .

push_xgboost_scikit_learn: build_xgboost_scikit_learn
	docker push us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:latest

build_local_image_onnx:
	docker buildx build --tag us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx:latest -f docker-image-onnx/Dockerfile .

build_image_onnx:
	docker buildx build --platform linux/amd64 --tag us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx:latest -f docker-image-onnx/Dockerfile .

push_image_onnx: build_image_onnx
	docker push us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx:latest
