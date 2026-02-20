# Patch typing.py to remove JAX dependency (use plain Python types)
source /root/octo_env/bin/activate

# Replace typing.py with JAX-free version
cat > /root/octo-pytorch/octo/utils/typing.py << 'ENDOFFILE'
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
ENDOFFILE
echo "TYPING_PATCH_DONE"

# Verify the patch works
python3 -c "from octo.utils.typing import Config, Params; print('typing import OK')" 2>&1; echo "TYPING_VERIFY_DONE"

# Now try the full import chain
python3 -c "from octo.model.octo_model_pt import OctoModelPt; print('OctoModelPt import OK')" 2>&1; echo "OCTO_IMPORT_DONE"
