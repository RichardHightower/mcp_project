from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import os

nest_asyncio.apply()

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key loaded: {'✓' if api_key else '✗'}")

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session = None
        self.anthropic = Anthropic()
        self.available_tools = []

    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]
        response = self.anthropic.messages.create(
            max_tokens = 2024,
            model = 'claude-3-7-sonnet-20250219',
            # tools exposed to the LLM
            tools = self.available_tools, 
            messages = messages
        )

        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        process_query = False
                elif content.type == 'tool_use':
                    assistant_content.append(content)
                    messages.append({'role':'assistant', 
                                   'content':assistant_content})
                    tool_id = content.id
                    tool_args = content.input
                    tool_name = content.name

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # Call a tool through the client session
                    result = await self.session.call_tool(tool_name, 
                                              arguments=tool_args)
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.content
                            }
                        ]
                    })

                    response = self.anthropic.messages.create(
                        max_tokens = 2024,
                        model = 'claude-3-7-sonnet-20250219',
                        tools = self.available_tools,
                        messages = messages
                    )

                    if len(response.content) == 1 and response.content[0].type == "text":
                        print(response.content[0].text)
                        process_query = False
                        


    # Add this method to your MCP_ChatBot class to handle
    # resources
    # Add this method to your MCP_ChatBot class to handle resources

    async def handle_resource(self, query):
        if query.startswith('@'):
            # Handle resource URI
            resource_name = query[1:]  # Remove the @ symbol
            try:
                # Fetch the resource content
                result = await self.session.get_resource(f"papers://{resource_name}")
                print(result.content)
                return True
            except Exception as e:
                print(f"Error accessing resource: {e}")
                return True
        return False

    # Add this method to handle prompts
    async def handle_prompt(self, query):
        if query.startswith('/prompt'):
            parts = query.split()
            if len(parts) == 1:
                # List available prompts
                response = await self.session.list_prompts()
                prompts = response.prompts
                print("\nAvailable prompts:")
                for prompt in prompts:
                    print(f"- {prompt.name}: {prompt.description}")
                    if prompt.parameters:
                        print("  Parameters:")
                        for param in prompt.parameters:
                            print(
                                f"    - {param.name}: {param.description} ({'optional' if not param.required else 'required'})")
                return True
            else:
                # Execute a specific prompt
                prompt_name = parts[1]
                # Parse parameters (format: key=value)
                params = {}
                for part in parts[2:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        # Convert to int if possible
                        if value.isdigit():
                            params[key] = int(value)
                        else:
                            params[key] = value

                result = await self.session.get_prompt(prompt_name, arguments=params)
                await self.process_query(result.content)
                return True
        return False

    # Update the chat_loop method to include handling for resources and prompts
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Special commands:")
        print("  @folders - List all available topic folders")
        print("  @<topic> - Get papers on a specific topic")
        print("  /prompt - List available prompts")
        print("  /prompt <name> <param=value> - Execute a specific prompt")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                # Check for special commands
                if await self.handle_resource(query):
                    continue

                if await self.handle_prompt(query):
                    continue

                # Process normal query
                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")



    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Command line arguments
            env=None,  # Optional environment variables
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", 
                                   [tool.name for tool in tools])

                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]

                await self.chat_loop()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())

