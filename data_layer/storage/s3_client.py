"""
S3 storage client with ETag support, conditional writes, and multipart uploads.

Supports both MinIO (via Tailscale) and AWS S3.
"""

import os
import json
from typing import Optional, Dict, List, BinaryIO
from pathlib import Path
from datetime import datetime

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger


class S3Client:
    """
    S3 client wrapper with MinIO/AWS compatibility.

    Provides:
    - ETag-based conditional operations (If-Match, If-None-Match)
    - Multipart upload support
    - Path-style addressing for MinIO
    - Consistent error handling
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        bucket: Optional[str] = None,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        force_path_style: bool = True,
    ):
        """
        Initialize S3 client.

        Args:
            endpoint_url: S3 endpoint (e.g., http://minio:9000)
            bucket: Default bucket name
            region: AWS region
            access_key: AWS/MinIO access key
            secret_key: AWS/MinIO secret key
            force_path_style: Use path-style addressing (required for MinIO)
        """
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT")
        self.bucket = bucket or os.getenv("S3_BUCKET", "ata")
        self.region = region or os.getenv("S3_REGION", "us-east-1")

        # Configure boto3 for MinIO compatibility
        config = Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path' if force_path_style else 'virtual'}
        )

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            region_name=self.region,
            aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=config
        )

        logger.info(f"S3Client initialized: endpoint={self.endpoint_url}, bucket={self.bucket}")

    def put_object(
        self,
        key: str,
        data: bytes,
        if_none_match: Optional[str] = None,
        if_match: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        bucket: Optional[str] = None,
    ) -> str:
        """
        Upload object to S3 with optional conditional write.

        Args:
            key: Object key (path)
            data: Object data as bytes
            if_none_match: Only write if ETag doesn't match (for create-only)
            if_match: Only write if ETag matches (for update-only)
            metadata: Custom metadata dict
            bucket: Override default bucket

        Returns:
            ETag of uploaded object

        Raises:
            ClientError: If conditional write fails or other S3 error
        """
        bucket = bucket or self.bucket

        kwargs = {
            'Bucket': bucket,
            'Key': key,
            'Body': data,
        }

        if if_none_match:
            kwargs['IfNoneMatch'] = if_none_match
        if if_match:
            kwargs['IfMatch'] = if_match
        if metadata:
            kwargs['Metadata'] = metadata

        try:
            response = self.s3.put_object(**kwargs)
            etag = response['ETag'].strip('"')
            logger.debug(f"Put object: s3://{bucket}/{key} (ETag: {etag})")
            return etag
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'PreconditionFailed':
                logger.warning(f"Conditional write failed for s3://{bucket}/{key}: {e}")
            raise

    def get_object(
        self,
        key: str,
        bucket: Optional[str] = None,
        if_match: Optional[str] = None,
    ) -> tuple[bytes, str]:
        """
        Download object from S3.

        Args:
            key: Object key
            bucket: Override default bucket
            if_match: Only get if ETag matches

        Returns:
            Tuple of (data, etag)

        Raises:
            ClientError: If object not found or conditional get fails
        """
        bucket = bucket or self.bucket

        kwargs = {'Bucket': bucket, 'Key': key}
        if if_match:
            kwargs['IfMatch'] = if_match

        try:
            response = self.s3.get_object(**kwargs)
            data = response['Body'].read()
            etag = response['ETag'].strip('"')
            logger.debug(f"Got object: s3://{bucket}/{key} ({len(data)} bytes)")
            return data, etag
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'NoSuchKey':
                logger.debug(f"Object not found: s3://{bucket}/{key}")
            raise

    def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        delimiter: Optional[str] = None,
    ) -> List[Dict]:
        """
        List objects with given prefix.

        Args:
            prefix: Key prefix to filter
            bucket: Override default bucket
            delimiter: Delimiter for hierarchical listing (e.g., '/')

        Returns:
            List of object metadata dicts with keys: Key, Size, ETag, LastModified
        """
        bucket = bucket or self.bucket

        kwargs = {'Bucket': bucket, 'Prefix': prefix}
        if delimiter:
            kwargs['Delimiter'] = delimiter

        objects = []
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(**kwargs):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'Key': obj['Key'],
                            'Size': obj['Size'],
                            'ETag': obj['ETag'].strip('"'),
                            'LastModified': obj['LastModified'],
                        })
            logger.debug(f"Listed {len(objects)} objects with prefix: s3://{bucket}/{prefix}")
        except ClientError as e:
            logger.error(f"List objects failed: {e}")
            raise

        return objects

    def delete_object(self, key: str, bucket: Optional[str] = None):
        """Delete object from S3."""
        bucket = bucket or self.bucket
        try:
            self.s3.delete_object(Bucket=bucket, Key=key)
            logger.debug(f"Deleted object: s3://{bucket}/{key}")
        except ClientError as e:
            logger.error(f"Delete failed for s3://{bucket}/{key}: {e}")
            raise

    def head_object(self, key: str, bucket: Optional[str] = None) -> Optional[Dict]:
        """
        Get object metadata without downloading.

        Returns:
            Metadata dict with ETag, Size, LastModified, or None if not found
        """
        bucket = bucket or self.bucket
        try:
            response = self.s3.head_object(Bucket=bucket, Key=key)
            return {
                'ETag': response['ETag'].strip('"'),
                'Size': response['ContentLength'],
                'LastModified': response['LastModified'],
                'Metadata': response.get('Metadata', {}),
            }
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return None
            raise

    def object_exists(self, key: str, bucket: Optional[str] = None) -> bool:
        """Check if object exists."""
        return self.head_object(key, bucket) is not None

    def put_json(
        self,
        key: str,
        data: dict,
        if_none_match: Optional[str] = None,
        if_match: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> str:
        """Upload JSON object."""
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        return self.put_object(
            key=key,
            data=json_bytes,
            if_none_match=if_none_match,
            if_match=if_match,
            metadata={'Content-Type': 'application/json'},
            bucket=bucket,
        )

    def get_json(self, key: str, bucket: Optional[str] = None) -> tuple[dict, str]:
        """Download and parse JSON object."""
        data, etag = self.get_object(key, bucket)
        return json.loads(data.decode('utf-8')), etag

    def multipart_upload(
        self,
        key: str,
        file_path: Path,
        part_size_mb: int = 5,
        bucket: Optional[str] = None,
    ) -> str:
        """
        Upload large file using multipart upload.

        Args:
            key: Object key
            file_path: Path to file to upload
            part_size_mb: Part size in MB (min 5MB for AWS S3)
            bucket: Override default bucket

        Returns:
            ETag of uploaded object
        """
        bucket = bucket or self.bucket
        part_size = part_size_mb * 1024 * 1024

        try:
            # Initiate multipart upload
            mpu = self.s3.create_multipart_upload(Bucket=bucket, Key=key)
            upload_id = mpu['UploadId']
            parts = []

            with open(file_path, 'rb') as f:
                part_num = 1
                while True:
                    data = f.read(part_size)
                    if not data:
                        break

                    # Upload part
                    response = self.s3.upload_part(
                        Bucket=bucket,
                        Key=key,
                        PartNumber=part_num,
                        UploadId=upload_id,
                        Body=data,
                    )

                    parts.append({
                        'PartNumber': part_num,
                        'ETag': response['ETag'],
                    })
                    part_num += 1

            # Complete multipart upload
            response = self.s3.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts},
            )

            etag = response['ETag'].strip('"')
            logger.info(f"Multipart upload complete: s3://{bucket}/{key} ({part_num-1} parts)")
            return etag

        except Exception as e:
            # Abort multipart upload on failure
            logger.error(f"Multipart upload failed, aborting: {e}")
            if 'upload_id' in locals():
                self.s3.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            raise


def get_s3_client() -> S3Client:
    """Factory function to create S3 client from environment."""
    return S3Client(
        endpoint_url=os.getenv("S3_ENDPOINT"),
        bucket=os.getenv("S3_BUCKET", "ata"),
        region=os.getenv("S3_REGION", "us-east-1"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        force_path_style=os.getenv("S3_FORCE_PATH_STYLE", "true").lower() == "true",
    )
