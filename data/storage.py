import glob
import boto3
import pandas as pd
import fnmatch
import io
from os import path
from pathlib import Path
import pickle
from decouple import config
from core.logs import logger


class LocalFileSystem:
    def __init__(self, path: str = "inputs_common/by_symbol/"):
        self.path = path

    def ls(self, symbol, pattern):
        """
        List files in the local directory that match the given pattern.
        """
        if symbol == '':
            symbol = '**'
        full_pattern = path.join(self.path, symbol, pattern)
        return sorted(glob.glob(full_pattern))

    @staticmethod
    def read_pickle(file='the actual filename, anything that comes after the symbol folder.pickle'):
        """
        Read a pickle file from the local filesystem.
        reads file without changing to add folders
        """
        return pd.read_pickle(file)

    def write_csv(self, data, symbol, file, **kwargs):
        """
        Write a DataFrame to a CSV file on the local filesystem.
        """
        file_path = self.filepath(symbol, file)
        data.to_csv(file_path, **kwargs)

    def write_excel(self, data, symbol, file, **kwargs):
        """
        Write a DataFrame to an Excel file on the local filesystem.
        """
        file_path = self.filepath(symbol, file)
        data.to_excel(file_path, **kwargs)

    def write_pickle(self, data, symbol, file):
        """
        Write a DataFrame to a pickle file on the local filesystem.
        """
        file_path = self.filepath(symbol, file)
        with open(file_path, 'wb') as handle:
            pickle.dump(data, handle)

    def filepath(self, symbol, file, create_parents=True):
        """
        Construct the file path for a given symbol and file name.
        """
        file_path = path.join(self.path, symbol, file)
        if not create_parents:
            return file_path
        parent = Path(file).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        return file_path


class WasabiStorageSystem:
    def __init__(self, path: str = "gex-signals/by_symbol", bucket: str | None = None):
        self.path = path
        # region_name = os.environ.get('AWS_REGION', 'us-east-2')
        host = config('WASABI_HOST', default='')
        access = config('WASABI_ACCESS_KEY_ID', default='')
        secret = config('WASABI_SECRET_ACCESS_KEY', default='')
        self.bucket_name = config('WASABI_BUCKET', default='nufintech-data-analysis')
        if bucket is not None:
            self.bucket_name = bucket
        if host.startswith('https://'):
            endpoint_url = host
        else:
            endpoint_url = 'https://' + host
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access,
            aws_secret_access_key=secret
        )

    @staticmethod
    def s3_all_keys(s3, bucket: str, prefix: str, delimiter: str | None = '/'):
        """Get a list of all keys in an S3 bucket."""
        keys = []
        if delimiter is not None:
            kwargs = {'Bucket': bucket, 'Prefix': prefix, 'Delimiter': delimiter}
        else:
            kwargs = {'Bucket': bucket, 'Prefix': prefix}
        while True:
            resp = s3.list_objects_v2(**kwargs)
            if 'Contents' not in resp:
                break

            for obj in resp['Contents']:
                keys.append(obj['Key'])

            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break

        return keys

    def ls(self, symbol, pattern):
        '''
        Returns the available keys for a pattern
        '''
        print(f"listing files matching {pattern}")
        prefix = path.join(self.path, symbol, '')  # '' is needed to add a '/' at the end
        if symbol == '':
            keys = self.s3_all_keys(self.client, self.bucket_name, prefix, None)
        else:
            keys = self.s3_all_keys(self.client, self.bucket_name, prefix)
        # for key in keys:
        #     if 'rev4' in key:
        #         print(key)
        full_pattern = path.join(self.path, symbol, pattern)
        logger.info(f"Searching for {full_pattern} in keys")
        selected = [key for key in keys if fnmatch.fnmatch(key, full_pattern)]
        return sorted(selected)

    def read_pickle(self, file):
        obj = self.client.get_object(Bucket=self.bucket_name, Key=file)
        df = pd.read_pickle(obj['Body'])
        return df

    def read_csv(self, file, **kwargs):
        obj = self.client.get_object(Bucket=self.bucket_name, Key=file)
        df = pd.read_csv(obj['Body'], **kwargs)
        return df

    def write_csv(self, data, symbol, file, **kwargs):
        key = self.filepath(symbol, file)
        with io.BytesIO() as buffer:
            data.to_csv(buffer, **kwargs)
            buffer.seek(0)  # Move cursor to start of the StringIO object
            # Upload CSV file to S3 bucket
            self.write(key, buffer)

    def write_excel(self, data, symbol, file, **kwargs):
        key = self.filepath(symbol, file)
        with io.BytesIO() as buffer:
            data.to_excel(buffer, **kwargs)
            buffer.seek(0)  # Move cursor to start of the StringIO object
            # Upload CSV file to S3 bucket
            self.write(key, buffer)

    def write_pickle(self, data, symbol, file):
        key = self.filepath(symbol, file)
        with io.BytesIO() as buffer:
            pickle.dump(data, buffer)
            buffer.seek(0)  # Move cursor to start of the StringIO object
            # Upload CSV file to S3 bucket
            self.write(key, buffer)

    # def filepath(self, symbol, file):
    #     return path.join(self.path, symbol, file)

    # adding this for widows
    def filepath(self, symbol, file):
        key = path.join(self.path, symbol, file)
        return key.replace("\\", "/")

    def write(self, key, buffer):
        self.client.upload_fileobj(buffer, self.bucket_name, key)


class AWSStorageSystem(WasabiStorageSystem):
    def __init__(self, path: str = "inputs_common/by_symbol", bucket: str | None = None):
        self.path = path
        # region_name = os.environ.get('AWS_REGION', 'us-east-2')
        host = config('AWS_REGION', default='us-east-1')
        access = config('AWS_ACCESS_KEY_ID', default='')
        secret = config('AWS_SECRET_ACCESS_KEY', default='')
        self.bucket_name = config('AWS_BUCKET', default='oaie')
        if bucket is not None:
            self.bucket_name = bucket
        self.client = boto3.client(
            's3',
            region_name=host,
            aws_access_key_id=access,
            aws_secret_access_key=secret
        )
