import re
import subprocess
import logging
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Ollama LLM
logging.debug("Initializing OllamaLLM with model 'codellama'")
llm = OllamaLLM(model="codellama")

# Define a strict prompt template to force only a command as output
prompt = PromptTemplate(
    input_variables=["query"],
    template="You are a Kubernetes CLI assistant. Respond with ONLY the exact kubectl command for: {query}"
)

# Create the pipeline
chain = prompt | llm

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

# Get user input dynamically
while True:
    query = input("\nAsk for a Kubernetes command (or type 'exit' to quit): ")
    logging.debug(f"User query: {query}")
    
    if query.lower() == "exit":
        print("Goodbye! 👋")
        logging.debug("Exiting the program.")
        break  # Exit the loop

    # Generate the kubectl command using CodeLlama
    logging.debug("Generating kubectl command using LangChain pipeline.")
    response = chain.invoke({"query": query})

    # Extract the first command inside triple quotes using regex
    match = re.search(r'```(?:sh|bash)?\n(.*?)\n```', response, re.DOTALL)
    
    if match:
        command = match.group(1).strip()  # Get the first command
        command = command.lstrip('$').strip()  # Remove leading '$'
        logging.debug(f"Extracted command: {command}")

        # Check if it's a command to execute locally (e.g., kubectl get pods)
        if "kubectl" in command:
            logging.debug("The command is related to kubectl, executing it locally.")
            # Execute the command on the local cluster and print the output
            local_output = execute_kubectl_command(command)
            print("\n🔹 Command Output from Local Cluster:\n", local_output)
        else:
            logging.debug(f"Command is not related to kubectl, just returning the command: {command}")
    else:
        logging.warning("No command found in response.")
        print("\n⚠️ No command found in response.")
