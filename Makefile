build:
	docker buildx build --tag us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:latest -f docker-xgboost-scikit-learn/Dockerfile .

push: build
	docker push us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:latest
