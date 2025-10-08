# Controlled Retrieval-augmented Context Evaluation for Long-form RAG

### Update
- 2025-10-06: Release data for crux-mds-duc04.
- TBD: data for crux-mds-multi_news. 
- TBD: evaluation function.

### Preparation
- Download datasets
```shell
cd /your_datasets/
git lfs install
git clone https://huggingface.co/datasets/DylanJHJ/crux
export CRUX_ROOT=/your_datasets/crux
```

- Install crux loading from source (v0.4.0)
We recommend to use the conda environment.
```shell
conda install -f environment.yaml
```
Then install crux from source:
```shell
git clone https://github.com/DylanJoo/crux
cd crux
uv pip install -e .
```

### Data loading 
We build the dataset dependent script to unify all
```python
os.environ["CRUX_ROOT"] = "/your_datasets/crux/"
from crux.tools.mds import load_data
data = load_data(subset="duc04")  # or "subet=multi_news"

from crux.tools.neuclir import load_data
data = load_data()
```

### Evaluation
```python
TBD
```

### Example of the first example in DUC04 test set
```json
{
  "id": "duc04-test-0",
  "topic": "Prepare a report on the violence and intimidation faced by abortion ...",
  "subtopics": [
      "Who was the mainstay of Buffalo"s only abortion clinic that was slain?",
      "What is the FBI looking for James Kopp for?",
      ...,
      "Who is Rev. Norman Weslin, and what is his role in the anti-abortion movement?",
      "What imperils the widespread availability of abortion procedures besides anti-abortion violence?"
  ],
    "report": "Dr. Barnett Slepian, the mainstay of Buffalo"s only abortion clinic, ...",
    "qrel": {
        "duc04-test-0:0#0": 1.0,
        "duc04-test-0:0#2": 1.0,
        ...
        "duc04-test-0:18#56": 1.0
    }
```
