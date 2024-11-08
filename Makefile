build:
	docker buildx build --tag us-west1-docker.pkg.dev/shiftsmart-api/orient-express-inference-xgboost-scikit-learn:latest -f docker-xgboost-scikit-learn/Dockerfile .
