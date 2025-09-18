# Controlled Retrieval-augmented Context Evaluation for Long-form RAG

Update: 
- 2025-09-18: We are now including other new datasets in addition to DUC04 and Multi-News. 
The data and the crux python package will be released soon.


### Installation
- Install crux from source (Beta version)
```shell
git clone https://github.com/DylanJoo/crux
cd crux
uv pip install -e .
```
- Prerequisite (recommend to use container + python venv)
TBD

### Example
```json
{
  "id": "example_id",
  "topic": "What is the impact of climate change on polar bears?",
  "questions": [
    "How does climate change affect polar bear habitats?",
    "What are the main threats to polar bears due to climate change?"
    ... (more questions)
  ],
  "passages": [
    {
      "contents": "Climate change is causing the Arctic ice to melt, which is crucial for polar bears.",
      "rating": 5
    },
    {
      "text": "Rising temperatures are leading to habitat loss for polar bears.",
      "rating": 4
    }
  ]
}
```
 

