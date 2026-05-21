XGBOOST_SCIKIT_LEARN := us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn
XGBOOST_TAG := v2.4.1

IMAGE_ONNX := us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx
IMAGE_ONNX_TAG := v2.4.1

GOOGLE_CLOUD_PROJECT ?=
REGION ?= us-west1
MODEL_NAME ?=
IMAGE_URL ?=
CONFIDENCE ?= 0.5

# Looked up from Vertex AI Model Registry by MODEL_NAME if not provided.
AIP_STORAGE_URI ?= $(shell gcloud ai models list \
	--region=$(REGION) --project=$(GOOGLE_CLOUD_PROJECT) \
	--filter='displayName=$(MODEL_NAME)' \
	--sort-by='~createTime' --limit=1 \
	--format='value(name)' | \
	xargs -I {} gcloud ai models describe {} \
	--region=$(REGION) --project=$(GOOGLE_CLOUD_PROJECT) \
	--format='value(artifactUri)')

print_aip_storage_uri:
	@echo "REGION=$(REGION)"
	@echo "GOOGLE_CLOUD_PROJECT=$(GOOGLE_CLOUD_PROJECT)"
	@echo "MODEL_NAME=$(MODEL_NAME)"
	@echo "--- step 1: model resource name ---"
	@gcloud ai models list \
		--region=$(REGION) --project=$(GOOGLE_CLOUD_PROJECT) \
		--filter='displayName=$(MODEL_NAME)' \
		--sort-by='~createTime' --limit=1 \
		--format='value(name)'
	@echo "--- step 2: artifact URI ---"
	@gcloud ai models list \
		--region=$(REGION) --project=$(GOOGLE_CLOUD_PROJECT) \
		--filter='displayName=$(MODEL_NAME)' \
		--sort-by='~createTime' --limit=1 \
		--format='value(name)' | \
	xargs -I {} gcloud ai models describe {} \
		--region=$(REGION) --project=$(GOOGLE_CLOUD_PROJECT) \
		--format='value(artifactUri)'
	@echo "--- resolved AIP_STORAGE_URI ---"
	@echo $(AIP_STORAGE_URI)

build_local_xgboost_scikit_learn:
	docker buildx build --tag $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG)_local -f docker-xgboost-scikit-learn/Dockerfile .

build_xgboost_scikit_learn:
	docker buildx build --platform linux/amd64 --tag $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG) -f docker-xgboost-scikit-learn/Dockerfile .

push_xgboost_scikit_learn: build_xgboost_scikit_learn
	docker push $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG)

run_local_xgboost_scikit_learn:
	docker run --rm \
		-v ~/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json:ro \
		-e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
		-e GOOGLE_CLOUD_PROJECT=$(GOOGLE_CLOUD_PROJECT) \
		-e AIP_STORAGE_URI=$(AIP_STORAGE_URI) \
		-e MODEL_NAME=$(MODEL_NAME) \
		-p 8080:8080 $(XGBOOST_SCIKIT_LEARN):$(XGBOOST_TAG)_local

build_local_image_onnx:
	docker buildx build --tag $(IMAGE_ONNX):$(IMAGE_ONNX_TAG)_local -f docker-image-onnx/Dockerfile .

build_image_onnx:
	docker buildx build --platform linux/amd64 --tag $(IMAGE_ONNX):$(IMAGE_ONNX_TAG) -f docker-image-onnx/Dockerfile .

push_image_onnx: build_image_onnx
	docker push $(IMAGE_ONNX):$(IMAGE_ONNX_TAG)

run_local_image_onnx:
	docker run --rm \
		-v ~/.config/gcloud/application_default_credentials.json:/root/.config/gcloud/application_default_credentials.json:ro \
		-e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
		-e GOOGLE_CLOUD_PROJECT=$(GOOGLE_CLOUD_PROJECT) \
		-e AIP_STORAGE_URI=$(AIP_STORAGE_URI) \
		-e MODEL_NAME=$(MODEL_NAME) \
		-p 8080:8080 $(IMAGE_ONNX):$(IMAGE_ONNX_TAG)_local

test_image_onnx:
	mkdir -p test-output
	@RESP=$$(mktemp); \
	trap "rm -f $$RESP" EXIT; \
	curl -s -X POST http://localhost:8080/v1/models/$(MODEL_NAME):predict \
		-H "Content-Type: application/json" \
		-d '{"instances": [{"image": "$(IMAGE_URL)"}], "parameters": {"confidence": $(CONFIDENCE)}}' \
		-o $$RESP && \
	jq -r '.predictions[0].debug_image // empty' $$RESP | base64 -d > test-output/debug_image.jpg && \
	jq -r '.predictions[0].predictions.class_mask // empty' $$RESP | base64 -d > test-output/class_mask.png && \
	jq -r '.predictions[0].predictions.valid_mask // empty' $$RESP | base64 -d > test-output/valid_mask.png && \
	jq 'del(.predictions[].debug_image?, .predictions[].predictions.class_mask?, .predictions[].predictions.valid_mask?)' $$RESP > test-output/response.json
