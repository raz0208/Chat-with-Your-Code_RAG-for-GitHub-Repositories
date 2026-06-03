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
    