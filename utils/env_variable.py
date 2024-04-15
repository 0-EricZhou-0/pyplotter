import os, functools

IS_DEBUG_ENVIORN = "DEBUG"
LOGGER_LEVEL = "DEBUG"

def check_env(env: str) -> None | str:
    return None if env not in os.environ else os.environ[env]

def check_env_exists(env: str) -> bool:
    return check_env(env) is not None

@functools.cache
def is_debug() -> bool:
    return check_env_exists(IS_DEBUG_ENVIORN)
