from typing import Any, Mapping, Sequence, Union, Tuple

try:
    import jax
    PRNGKey = jax.random.KeyArray if hasattr(jax.random, 'KeyArray') else type(jax.random.PRNGKey(0))
    PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
    Dtype = jax.typing.DTypeLike
except Exception:
    PRNGKey = Any
    PyTree = Union[Any, Mapping[str, "PyTree"]]
    Dtype = Any

Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
