#!/usr/bin/env python3
"""Test GitHub token authentication."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
print(f"Token: {token[:10]}...")
print(f"Token length: {len(token)}")

# Test with simple query
query = """
{
  viewer {
    login
  }
}
"""

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("\nTesting GitHub GraphQL API...")
response = requests.post(
    "https://api.github.com/graphql",
    headers=headers,
    json={"query": query},
    timeout=30
)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text[:500]}")

# Try with token in different format
headers2 = {
    "Authorization": f"token {token}",
    "Content-Type": "application/json"
}

print("\nTrying with 'token' prefix instead of 'Bearer'...")
response2 = requests.post(
    "https://api.github.com/graphql",
    headers=headers2,
    json={"query": query},
    timeout=30
)

print(f"Status code: {response2.status_code}")
print(f"Response: {response2.text[:500]}")
