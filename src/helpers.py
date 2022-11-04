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
