# Controlled Retrieval-augmented Context Evaluation for Long-form RAG

### Update
- 2025-10-06: Release data for crux-mds-duc04
- 2025-10-08: Release data for crux-mds-multi_news
- 2025-10-08: Release data for crux-neuclir
- 2025-10-21: Release reproducible evaluation result on DUC04. See [runs](runs/)
- TBD: evaluation function.

### Preparation
- Download datasets and the corpus
```shell
cd your_datasets/
git lfs install
git clone https://huggingface.co/datasets/DylanJHJ/crux
git clone https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus
```

- Installation 
Install crux from source:
```shell
git clone https://github.com/DylanJoo/crux
cd crux
pip install -e .
```
[IMPORTANT] Set CRUX_ROOT environment variable to the path where you downloaded the datasets.
```shell
export CRUX_ROOT=/your_datasets/crux
```
OR
```python
os.environ["CRUX_ROOT"] = "/your_datasets/crux/"
```

### Data loading 
We have provided data loading functions in crux. Currently, we support crux-mds-duc04 and crux-mds-multi_news, and neuclir.

See the data format below. Full content example is listed at the end of this README.
```python
from crux.tools.mds.ir_utils import load_data
data = load_data(subset="duc04")  # or "subet=multi_news"
print(data.iloc[0])

>>> 
topic        Prepare a report on the violence and intimidat...
subtopics    [Who was the mainstay of Buffalo's only aborti...
report       Dr. Barnett Slepian, the mainstay of Buffalo's...
qrel         {'duc04-test-0:0#0': 1.0, 'duc04-test-0:0#2': ...
Name: duc04-test-0, dtype: object
```

For NeuCLIR, we leave the `report` field empty, as it is not provided in the original dataset.
```python
from crux.tools.neuclir.ir_utils import load_data
data = load_data()
print(data.iloc[0])

>>> 
topic        Japan suicide rate COVID-19 I need a report on...
subtopics    [[How many years has it been since Japan had t...
report                        No ground-truth report provided.
qrel         {'ba30498c-9dbf-4b1d-bbfa-bcdca4548b18': 3.0, ...
Name: 300, dtype: object
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
