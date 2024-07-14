import os
from typing import get_type_hints, Any

from dotenv.main import load_dotenv

load_dotenv()


class AppConfigError(Exception):
    pass


def eval_bool(value: str) -> bool:
    """
    Function used to parse boolean values in env variables
    :param value: string value of the env variable
    :return: boolean interpretation of the value
    """
    try:
        if value.upper() == "TRUE":
            return True
        else:
            return False
    except AttributeError:
        # Empty variable
        return False


# AppConfig class with required fields, default values, type checking, and typecasting for int and bool values
class Config:
    """
    Map environment variables to class fields according to these rules:
      - Field won't be parsed unless it has a type annotation
      - Class field and environment variable name are the same
    """

    POSTGRES_DB_HOST: str
    POSTGRES_DB_NAME: str
    POSTGRES_DB_USER: str
    POSTGRES_DB_PASSWORD: str
    POSTGRES_DB: str

    @staticmethod
    def get_value(value: str) -> Any:
        # Cast env var value to expected type and raise AppConfigError on failure
        var_type = get_type_hints(Config)[value]
        try:
            if var_type == bool:
                value = eval_bool(os.environ.get(value))
            else:
                value = var_type(os.environ.get(value))
            return value

        except ValueError:
            raise AppConfigError(
                'Unable to cast value of to type "{}" for "{}" field'.format(
                    var_type, value
                )
            )

    def __repr__(self):
        return str(self.__dict__)
