import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

from vllm.engine.server import serve
import aiohttp

# Function to start the vLLM server
# async def start_vllm_server():
#     parser = ServerArgumentParser()
#     args = parser.parse_args([
#         "--model", "meta-llama/Llama-3.3-70B-Instruct",  # Replace with your desired model
#         "--pipeline-parallel-size", "4",  # Number of GPUs
#         "--port", "8000",  # The server port
#     ])
#     await serve(args)

async def start_vllm_server():
    parser = ServerArgumentParser()
    args = parser.parse_args([
        "--model", "meta-llama/Llama-3.3-70B-Instruct",  # Replace with your desired model
        "--pipeline-parallel-size", "4",  # Number of GPUs
        "--port", "8000",  # The server port
    ])
    await serve(args)

# Function to send asynchronous requests to the vLLM server
async def send_requests():
    async with aiohttp.ClientSession() as session:
        # Define the request payload
        payload = {
            "prompt": "Tell a 500 word story about Amsterdam.",
            "max_tokens": 1024,
            "min_tokens": 512,
        }

        # Send the request
        async with session.post("http://0.0.0.0:8000/v1", json=payload) as response:
            response_json = await response.json()
            print("Response:", response_json)

# Main function to run the server and make requests
async def main():
    # Run the server in a separate task
    server_task = asyncio.create_task(start_vllm_server())

    # Wait for the server to initialize (give it a few seconds to start)
    await asyncio.sleep(5)

    # Send requests to the server
    await send_requests()

    # Cancel the server task to shut it down
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

