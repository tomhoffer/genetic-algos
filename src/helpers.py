def eval_bool(value: str) -> bool:
    """
    Function used to parse boolean values in env variables
    :param value: string value of the env variable
    :return: boolean interpretation of the value
    """

    if value.upper() == "TRUE":
        return True
    else:
        return False
