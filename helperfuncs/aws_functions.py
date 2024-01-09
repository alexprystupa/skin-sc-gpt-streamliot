# aws_functions.py
import boto3
import io
import pandas as pd


def get_aws_S3_client(access_key, secret_key):
    return boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)


def read_S3_meta_data(s3_client):
    response = s3_client.get_object(Bucket='sc-pdf-recommendation-bucket',
                                    Key='data/meta-data/Sources-Titles-Meta-Data.csv')

    return pd.read_csv(io.BytesIO(response['Body'].read()))


def get_s3_first_img_path_all_papers(s3_client, bucket_name):
    img_path = []
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    files = response.get("Contents")

    for file in files:
        if file['Key'].endswith("page_1.jpg"):
            img_path.append(f"s3://{bucket_name}/{file['Key']}")

    return img_path


def get_s3_img_paths_recommended_paper(s3_client, bucket_name, pmid):
    image_s3_paths = []

    response = s3_client.list_objects_v2(Bucket=bucket_name)
    files = response.get("Contents")

    num_pages = len([file for file in files if pmid in file['Key']])

    for i in range(1, num_pages + 1):
        image_s3_paths.append(f"s3://{bucket_name}/data/pdf-images/PDF-IMG-{pmid}/page_{i}.jpg")

    return image_s3_paths
