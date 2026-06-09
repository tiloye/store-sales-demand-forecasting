ENV ?= dev
ENV_FILE := .env.$(ENV)
-include $(ENV_FILE)
export

start-dev-containers:
	docker compose -f dev-compose.yml up -d --wait

remove-dev-containers:
	docker compose -f dev-compose.yml down

restart-dev-containers:
	$(MAKE) remove-dev-containers
	$(MAKE) start-dev-containers

# TEST

test: start-dev-containers
	pytest tests/

start-astro-airflow-standalone:
	astro dev start --standalone

stop-astro-airflow-standalone:
	astro dev kill --standalone

run-astro-dev-test: start-garage start-astro-airflow-standalone
	astro dev pytest --standalone

start-local-mlflow-server:
	mlflow server \
	    --backend-store-uri sqlite:///mlflow.db --port 5000

create-astro-deployment-variables:
	astro deployment variable create \
		ENV_NAME=$(ENV_NAME) \
		MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		AWS_ENDPOINT_URL=$(AWS_ENDPOINT_URL) \
		AWS_REGION=$(AWS_REGION) \
		S3_BUCKET_NAME=$(S3_BUCKET_NAME) \
		EVIDENTLY_WORKSPACE_URL=$(EVIDENTLY_WORKSPACE_URL) \
		-d $(ASTRO_DEPLOYMENT_ID)

	astro deployment variable create \
		KAGGLE_API_TOKEN=$(KAGGLE_API_TOKEN) \
		AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
		AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		MLFLOW_TRACKING_USERNAME=$(MLFLOW_TRACKING_USERNAME) \
		MLFLOW_TRACKING_PASSWORD=$(MLFLOW_TRACKING_PASSWORD) \
		EVIDENTLY_API_KEY=$(EVIDENTLY_API_KEY) \
		EVIDENTLY_ORG_ID=$(EVIDENTLY_ORG_ID) \
		-d $(ASTRO_DEPLOYMENT_ID) \
		--secret
