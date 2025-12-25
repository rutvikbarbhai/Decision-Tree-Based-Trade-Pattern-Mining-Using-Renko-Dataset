from pydantic import Field  # , AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # PG_CONN: AnyUrl = Field('')
    POLYGON_SECRET_ACCESS_KEY: str = Field(default='')
    REDIS_HOST: str = Field(default='renko-optimizer-master-redis-leader.airflow.svc')
    REDIS_PORT: int = Field(default=6379)

    WASABI_ACCESS_KEY_ID: str = Field(default='')
    WASABI_SECRET_ACCESS_KEY: str = Field(default='')
    WASABI_HOST: str = Field(default='s3.us-east-2.wasabisys.com')
    WASABI_BUCKET: str = Field(default='nufintech-data-analysis')

    INDICATORS_URL: str = Field(default="https://indicators-v2.ewr4.data-processor.nufintech.com/")
    SPOTS_URL: str = Field(default="https://indicators-v2.ewr4.data-processor.nufintech.com/calculate")
