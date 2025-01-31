from openai import AsyncOpenAI
from datetime import datetime
import asyncio
import os
import subprocess
import time
import requests
import sys
import signal

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"
client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

PROMPT = """Your job is to generate 100 queries that the following passage is relevant to. The queries should be diverse in terms of level of granularity, generality, the number of entities and concepts in the passage, and whether they agree or disagree with the views present in the passage. Consider what values are possible for each of these dimensions, and then generate queries for different combinations of these values. The passage should be related to every query that you generate, and it should be possible for the queries to also be related to other passages. The queries should not be so highly tailored to this passage that it is very unlikely for the queries to be relevant to any other passages. All queries should contain only short phrases and/or keywords. All queries should be between 2 and 6 words.  Output the queries as a list of strings in JSON format.
Passage:
Before the year 2000 the clearest view of distant galaxies in the universe
will come from a European supertelescope in Chile's Atacama desert.
Eight countries - Germany, France, the Netherlands, Belgium, Sweden,
Denmark, Italy and Switzerland - are working together at the European
Southern Observatory (ESO) to try to surpass the achievements of the US
Hubble Space Telescope. And they hope to do this for a fraction of Hubble's
Dollars 2bn (Pounds 1bn) cost.
ESO's showcase is the Very Large Telescope (VLT) project. By 1999 four large
telescopes will work in unison to capture the equivalent light of a single
16-metre instrument. The VLT will be the world's most powerful telescope,
nearly three times larger than any operating today. This great eye is
expected to cost DM450m (Pounds 156m).
The VLT's progress is creating as much excitement among astronomers as the
launch of the Hubble nearly two years ago. The Hubble space telescope,
orbiting 300 miles above the Earth's atmosphere, was expected to reveal
uncharted expanses of the universe by 'seeing' seven times deeper into space
than observatories on Earth.
However, Hubble's mission has been partially crippled by microscopic
scratches on the surface of its 2.4-metre mirror. The images being relayed
back to Earth have failed to live up to their revolutionary promise.
Hubble's misfortunes have rekindled interest in a new generation of
Earth-bound telescopes and sharpened the rivalry between US and European
astronomers.
The Association of Universities for Research in Astronomy has obtained US
government backing for an eight-metre telescope in Hawaii. Two more 10-metre
telescopes in Hawaii are being financed by the private Keck Foundation.
The VLT, however, remains in a class of its own. It will allow astronomers
to explore three quarters of the universe and study galaxies perhaps as far
as 14bn light years away. The VLT will also be powerful enough to penetrate
the innermost regions of active galaxies, which may harbour black holes at
their centres. As a result astrophysicists will learn more about the
chemical composition of stars and interstellar clouds.
After gathering meteorological data for six years, ESO decided that Cerro
Paranal, 800km north of its existing 14-telescope observatory in La Silla,
was probably the best site in the world for astronomical work. Cerro Paranal
boasts perfect photometric nights for 60 per cent of the year and a very
still atmosphere. Astronomers are looking forward to unexpected discoveries.
'The universe has far more imagination than we have,' says Christoffel
Waelkens, a visiting Belgian astronomer.
The greatest technological challenge for scientists working on the VLT
project will be to cast four identical single-blank mirrors, each 8.2 metres
in diameter. Until recently this was thought to be impossible, because very
large mirrors buckle under their own weight.
But Schott Glaswerke in Germany has already cast the first 8.2-metre mirror
by pouring molten glass ceramic into a rotating concave mould. The spinning
action produced a much thinner and lighter mirror of greater width -
clearing a technological barrier that had limited astronomy for half a
century.
It is not the VLT's size that is the envy of US astronomers, but its
revolutionary image-focusing capability, called 'adaptive optics'. In this
state-of-the-art technology, 200 electronic arms gently push and pull the
telescope's mirror to eliminate the 'twinkling' of stars caused by
atmospheric turbulence. By bringing the fuzzy images into focus, the picture
quality is almost as sharp as if taken from space.
The first telescope to use 'adaptive optics' was inaugurated at La Silla in
1989. It is now capturing images that are several times sharper than those
obtained by telescopes of conventional design. 'From an optical point of
view, we have built the best telescope in the world today,' says Daniel
Hofstadt, La Silla's technical manager.
Hofstadt, however, does not see the VLT project or adaptive optics replacing
the need for telescopes in space. Terrestrial observatories, he explains,
cannot detect faint stars that emit ultraviolet light because this is
blotted out by the Earth's protective ozone layer. 'The VLT and the Hubble
will complement each other,' he says.
"""

def wait_for_server():
    health_url = "http://0.0.0.0:8000/health"
    max_retries = 30
    retry_delay = 30

    for _ in range(max_retries):
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                print("vLLM server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print("Waiting for vLLM server to start...")
        time.sleep(retry_delay)

    print("Failed to connect to vLLM server after max retries")
    return False

async def main():
    tasks = []
    for i in range(100):
        print("***submit", datetime.now())
        task = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": PROMPT}
            ]
        )
        tasks.append(task)

    completions = await asyncio.gather(*tasks)

    for completion in completions:
        print("Completion result:", completion)

def gpu_count():
   gpu_list = os.environ.get("SLURM_STEP_GPUS", os.environ['SLURM_JOB_GPUS'])
   return len(gpu_list.split(","))

if __name__ == "__main__":
    MY_GPU_COUNT = gpu_count()
    MAX_LEN = 8192
    cmd = f"python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --gpu-memory-utilization 0.95 --max-model-len {MAX_LEN} --tensor-parallel-size {MY_GPU_COUNT} --distributed-executor-backend=mp --enforce-eager --dtype bfloat16"
    server_process = subprocess.Popen(
        cmd.split(),
    )

    if not wait_for_server():
        server_process.terminate()
        sys.exit(1)

    try:
        print("***main start", datetime.now())
        asyncio.run(main())
        print("***main stop", datetime.now())
    finally:
        server_process.terminate()
        server_process.wait()
        print("Server process terminated")
