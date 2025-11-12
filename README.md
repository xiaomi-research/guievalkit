```shell
pip install uv

pushd ./guievalkit/
uv venv ur_venv
source ur_venv/bin/activate
uv pip install -r requirements.txt -i accessible_url  # uv won't read accessible url from pip.conf
```