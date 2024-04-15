import inspect, ast

def get_source_location(obj) -> str | None:
    try:
        source_file = inspect.getsourcefile(obj)
        source_line = inspect.getsourcelines(obj)[-1]
        return f"{source_file}:{source_line}"
    except:
        return None
