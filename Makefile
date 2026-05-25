ENV ?= dev
ENV_FILE := .env.$(ENV)
-include $(ENV_FILE)
export

start-garage:
	docker run -d \
	    --name garage-container \
	    -p 3900:3900 -p 3901:3901 -p 3902:3902 -p 3903:3903 \
	    -v $(PWD)/garage.toml:/etc/garage.toml \
	    -e GARAGE_DEFAULT_ACCESS_KEY=$(AWS_ACCESS_KEY_ID) \
	    -e GARAGE_DEFAULT_SECRET_KEY=$(AWS_SECRET_ACCESS_KEY) \
	    -e GARAGE_DEFAULT_BUCKET=$(S3_BUCKET_NAME) \
	    dxflrs/garage:v2.3.0 /garage server --single-node --default-bucket

stop-garage:
	docker stop garage-container || true
	docker rm garage-container || true

test: start-garage
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
		MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) \
		AWS_ENDPOINT_URL=$(AWS_ENDPOINT_URL) \
		AWS_REGION=$(AWS_REGION) \
		S3_BUCKET_NAME=$(S3_BUCKET_NAME) \
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
