docker run -d \
    --name garage-container \
    -p 3900:3900 -p 3901:3901 -p 3902:3902 -p 3903:3903 \
    -v $(pwd)/garage.toml:/etc/garage.toml \
    -e GARAGE_DEFAULT_ACCESS_KEY=$AWS_ACCESS_KEY_ID \
    -e GARAGE_DEFAULT_SECRET_KEY=$AWS_SECRET_ACCESS_KEY \
    -e GARAGE_DEFAULT_BUCKET=$S3_BUCKET_NAME \
    dxflrs/garage:v2.3.0 /garage server --single-node --default-bucket