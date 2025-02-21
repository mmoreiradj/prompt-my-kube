import subprocess
import logging
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import Optional, Type
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

# Import things that are needed generically
from langchain.tools import BaseTool

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

# Function to execute kubectl commands locally
def execute_kubectl_command(command: str):
    logging.debug(f"Executing kubectl command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        logging.debug(f"Command executed successfully. Output: {result.stdout.strip()}")
        return result.stdout.strip()  # Capture the standard output
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e.stderr.strip()}")
        return f"Error executing command: {e.stderr.strip()}"

# Tools


class GetResourcesInput(BaseModel):
    namespace: str = Field(description="should be a valid kubernetes namespace")

class GetResourcesTool(BaseTool):
    name: str = "get_resources"  # Explicitly define the type
    description: str = "Get all the resources in the kubernetes cluster in the specified namespace. If no namespace is specified use as default value 'default'"
    args_schema: Type[BaseModel] = GetResourcesInput

    def _run(
        self, namespace: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the kubectl command."""
        return execute_kubectl_command(f"kubectl get all -n {namespace}")

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Ollama LLM
logging.debug("Initializing OllamaLLM with model 'codellama'")
llm = OllamaLLM(model="codellama")

query = "get the pods running in the cluster"

# Define a strict prompt template to force only a command as output
prompt = hub.pull("hwchase17/react")

# Define a strict prompt template to force only a command as output
prompt_template = PromptTemplate.from_template(
    """
    You are a Kubernetes expert, you have to perform some tasks on a local kubernetes cluster. 
    This is the task you need to do {query} 
    """
)


getResourceTool = GetResourcesTool()

tools = [getResourceTool]

prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Give the resources running in the cluster in the namespace kube-system", "chat_history": []})

