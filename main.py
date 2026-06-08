# Import required libraries

# Apply nest_asyncio to allow nested event loops in Jupyter notebooks or similar environments.
import nest_asyncio 
nest_asyncio.apply()

import os
import re
# import textwarp
from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext

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
def initialize_github_client(token):
    github_token = os.getenv(token)
    return GithubClient(github_token)

# Check for OpenAI API key
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OpenAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Check for GitHub Token
GitHub_token = os.getenv("GITHUB_TOKEN")
if not GitHub_token:
    raise ValueError("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")

# Check for Deaplake API Token
deeplake_token = os.getenv("DEEPLAKE_TOKEN")
if not deeplake_token:
    raise ValueError("Deeplake API token not found. Please set the DEEPLAKE_TOKEN environment variable.")

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

# Initioalize the GitHub repository loader
loader = GithubRepositoryReader(
    gitHub_client = gitHub_client, # Ensure your cilent variable matched this
    owner = owner,
    repo = repo,
    filter_file_extensions = [".py", ".js", ".ts", ".md"],
    verbose = False,
    concurrent_requests = 5,
)

print(f"Loading {repo} repository by {owner}...")
docs = loader.load_data(branch = 'main')

print("Documents loaded: ")
for doc in docs:
    print(doc.metadata)

print("Uploading to vector store ...")
