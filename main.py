# Import required libraries

# Apply nest_asyncio to allow nested event loops in Jupyter notebooks or similar environments.
import nest_asyncio 
nest_asyncio.apply()

import os
import re
import textwarp
from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import download_loader, VectorStoreIndex
from llama_index.vector_stores.deeplake import DeeplakeVectorStore
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