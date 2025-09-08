# Data Preparation

## Android Control

Download [Android Control](https://github.com/google-research/google-research/tree/master/android_control) and save at ``guievalkit/data/android_control``.
```bash
pip3 install tensorflow android_env

cd guievalkit/data

python3 process_ac.py
```

## CAGUI (Agent)

```bash
cd guievalkit/data

mkdir cagui_agent && cd cagui_agent
huggingface-cli download openbmb/CAGUI --repo-type dataset --include "CAGUI_agent/**" --local-dir ./ --local-dir-use-symlinks False --resume-download
mv CAGUI_agent test
```

## AiTZ

Download [AiTZ](https://github.com/IMNearth/CoAT) and save at ``guievalkit/data/android_in_the_zoo``.

```bash
python3 process_aitz.py
```

## GUI Odyssey

```bash
huggingface-cli download --repo-type dataset --resume-download OpenGVLab/GUI-Odyssey --local-dir gui_odyssey
cp ./utils/preprocessing.py gui_odyssey
cp ./utils/format_converter.py gui_odyssey
cd gui_odyssey
python3 preprocessing.py
python3 format_converter.py
cd ..
python3 process_odyssey.py
```
