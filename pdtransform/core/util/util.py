import typing

T = typing.TypeVar('T')


def traverse(obj: T, callback: typing.Callable) -> T:
    if isinstance(obj, list):
        return [traverse(elem, callback) for elem in obj]
    if isinstance(obj, dict):
        return {k: traverse(v, callback) for k, v in obj.items()}
    if isinstance(obj, tuple):
        if type(obj) != tuple:
            # Named tuples cause problems
            raise Exception
        return tuple(traverse(elem, callback) for elem in obj)

    return callback(obj)
