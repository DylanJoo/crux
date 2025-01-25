import asyncio
from vllm.engine.arg_parser import ServerArgumentParser
from vllm.engine.server import serve
import aiohttp

# Function to start the vLLM server
async def start_vllm_server():
    parser = ServerArgumentParser()
    args = parser.parse_args([
        "--model", "gpt-3.5-turbo",  # Replace with your desired model
        "--port", "8000",  # The server port
    ])
    await serve(args)

# Function to send asynchronous requests to the vLLM server
async def send_requests():
    async with aiohttp.ClientSession() as session:
        # Define the request payload
        payload = {
            "prompt": "Explain asynchronous programming in Python.",
            "max_tokens": 50,
        }

        # Send the request
        async with session.post("http://127.0.0.1:8000/generate", json=payload) as response:
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

