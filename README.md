# Controlled Retrieval-augmented Context Evaluation for Long-form RAG

### Update
- 2025-10-06: Release data for crux-mds-duc04
- 2025-10-08: Release data for crux-mds-multi_news
- 2025-10-08: Release data for crux-neuclir
- 2025-10-21: Release evaluation script and result on DUC04. See [runs](runs/)
- TBD: evaluation function.

### Preparation
- Download [crux-data](https://huggingface.co/datasets/DylanJHJ/crux) and the [crux-mds-corpus](https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus).
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
We support the run file in `TREC` format. The evaluation implementation is 

```shell
cd crux

CRUX_ROOT=/your_datasets/crux
subset=crux-mds-duc04

python -m crux.evaluation.rac_eval \
    --run $run_file \
    --qrel ${CRUX_ROOT}/${subset}/qrels/div_qrels-tau3.txt \
    --filter_by_oracle \
    --judge ${CRUX_ROOT}/${subset}/judge/ratings.Llama-3.1-70B-Instruct.0-1.jsonl
>>>
2025-10-21 10:50:43,196 - INFO - Loading run/qrel with topk=10, threshold=1...
2025-10-21 10:50:43,219 - INFO - Loading run/qrel with topk=1000, threshold=1...
runs | bm25.default.crux-mds-duc04.txt | P@10 | 0.5140 | nDCG@10 | 0.5298 | alpha_nDCG@10 | 0.4454 | Cov@10 | 0.5444 | 
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
