# Import required libraries

# Import required libraries
import nest_asyncio 
nest_asyncio.apply()

import os
import re
import textwrap
from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import VectorStoreIndex, Settings  # Added Settings
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext

# Import Groq and HuggingFace wrappers
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the environment variables from your .env file
load_dotenv()

# Extract owner and repo from GitHub URL
def parse_github_url(url): 
    pattern = r'https://github\.com/([^/]+)/([^/]+)' #
    match = re.match(pattern, url)
    if match:
        return match.groups()
    else:
        raise ValueError("Invalid GitHub URL format. Expected format: https://github.com/<owner>/<repo>")
        return (None, None)

# Validate owner and repo    
def validate_owner_repo(owner, repo):
    return bool(owner), bool(repo)

# Initialize GitHub client
def initialize_github_client(token_env_name):
    github_token = os.getenv(token_env_name)
    return GithubClient(github_token)

# Fetch DATASET_PATH from environment variables
DATASET_PATH = os.getenv("DATASET_PATH")
if not DATASET_PATH:
    raise ValueError("DATASET_PATH not found. Please set the DATASET_PATH environment variable.")

# # Check for OpenAI API key if you have it
# OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OpenAI_API_KEY:
#     raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Verify all required API Keys are loaded
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Check for Groq API Key
if not GROQ_API_KEY:
    raise ValueError("Groq API key not found. Please set the GROQ_API_KEY in your .env file.")

GitHub_token = os.getenv("GITHUB_TOKEN") # Check for GitHub Token
if not GitHub_token:
    raise ValueError("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")

active_loop_token = os.getenv("ACTIVELOOP_TOKEN") # Check for Deaplake API Token
if not active_loop_token:
    raise ValueError("Deeplake API token not found. Please set the DEEPLAKE_TOKEN environment variable.")

# # Initialize Groq via OpenAI Client
# client = OpenAI( # Note: the correct argument name 'api_key' (lowercase)
#     api_key=GROQ_API_KEY,
#     base_url="https://api.groq.com/openai/v1",
# )

# # Corrected syntax for Groq/OpenAI Chat Completion
# print("Testing Groq API connection...")
# response = client.chat.completions.create(
#     model="openai/gpt-oss-20b", # Note: use a valid Groq model here, e.g., 'llama3-8b-8192' or 'mixtral-8x7b-32768'
#     messages=[
#         {"role": "user", "content": "Explain the importance of fast language models"}
#     ]
# )
# print("Groq Response:")
# print(response.choices[0].message.content)
# print("-" * 40)

# Initialize Github client
gitHub_client = initialize_github_client("GITHUB_TOKEN")
# download_loader(GithubRepositoryReader)

# Function to get and validate GitHub Repo URL
def get_valid_repo_data():
    while True:
        github_url = input("Please enter the GitHub repository URL: ")
        try:
            owner, repo = parse_github_url(github_url)
            if validate_owner_repo(owner, repo):
                return owner, repo
        except Exception as e:
            print(f"An error occured: {e}")
            pass
        
        print("Invalid GitHub URL. Please try again.\n")

# Call function to get valid GitHub URL as client input
owner, repo = get_valid_repo_data()

# Configure LlamaIndex to use Groq and HuggingFace
print("Configuring LLM and Embedding models...")
Settings.llm = Groq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

# Using a standard local embedding model from HuggingFace
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initioalize the GitHub repository loader
loader = GithubRepositoryReader(
    github_client = gitHub_client, # Ensure your cilent variable matched this
    owner = owner,
    repo = repo,
    filter_file_extensions = ([".py", ".js",".ts", ".md"], GithubRepositoryReader.FilterType.INCLUDE),
    verbose = False,
    concurrent_requests = 5,
)

print(f"Loading {repo} repository by {owner}...")
docs = loader.load_data(branch = 'main')

print("Documents loaded: ")
for doc in docs:
    print(doc.metadata)

print("Uploading to vector store ...")

# Create vectore store and upload data
vectore_store = DeepLakeVectorStore(
    dataset_path = DATASET_PATH,
    overwrite = True,
    runtime = {"tensor_db": True}
)

# Create storage context and index
storage_context = StorageContext.from_defaults(vector_store = vectore_store)
index = VectorStoreIndex.from_documents(docs, storage_context = storage_context)
query_engine = index.as_query_engine()

print("Data uploaded to vector store successfully!")

# A simple quation to test
intro_question = "What is the purpose of this repository?"
print(f"Test Question: {intro_question}")
print("=" *50)
answer = query_engine.query(intro_question)
print(f"Answer: {textwrap.fill(str(answer), 100)} \n")


# Interactive query loop
while True:
    user_query = input("Ask a question about the repository (or type 'exit' to exit): ")
    if user_query.lower() == 'exit':
        print("Exiting...")
        break
    
    print("=" *50)
    print(f"Your question: {user_query}")
    
    answer = query_engine.query(user_query)
    print(f"Answer: {textwrap.fill(str(answer), 100)} \n")