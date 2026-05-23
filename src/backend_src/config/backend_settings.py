from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


class BackendSettings(BaseSettings):
    API_HOST: str = os.getenv("API_HOST")
    API_PORT: int = os.getenv("API_PORT")

    class Config:
        env_file = ".env"
        extra = "allow"