import inspect
from chatnoir_api.v1 import search
from chatnoir_api import Index, cache_contents

api_key: str = "<API_KEY>"
results = search("python library", index="clueweb22/b")
print([r.uuid for r in results])
# print([ (r.uuid, r.score) for r in results])


contents = cache_contents(
    "clueweb09-en0051-90-00849",
    index="clueweb09",
)
print(contents)

plain_contents = cache_contents(
    "clueweb09-en0051-90-00849",
    index="clueweb09",
    plain=True,
)
print(plain_contents)
