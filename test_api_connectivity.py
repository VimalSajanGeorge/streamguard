"""Test API connectivity for CVE and GitHub collectors."""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_nvd_api():
    """Test NVD API connectivity."""
    print("\n" + "="*60)
    print("Testing NVD API")
    print("="*60)

    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {
        'keywordSearch': 'SQL injection',
        'resultsPerPage': 5,
        'startIndex': 0,
        'pubStartDate': '2024-08-15T00:00:00.000',
        'pubEndDate': '2024-12-12T23:59:59.999'
    }

    try:
        print(f"URL: {url}")
        print(f"Params: {params}")
        print("\nSending request...")

        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        response.raise_for_status()

        data = response.json()
        vulnerabilities = data.get('vulnerabilities', [])

        print(f"\n+ SUCCESS")
        print(f"  Total results: {data.get('totalResults', 0)}")
        print(f"  Results returned: {len(vulnerabilities)}")
        print(f"  Results per page: {data.get('resultsPerPage', 0)}")

        if vulnerabilities:
            first_cve = vulnerabilities[0].get('cve', {})
            print(f"\n  First CVE ID: {first_cve.get('id')}")
            descriptions = first_cve.get('descriptions', [])
            if descriptions:
                desc = descriptions[0].get('value', '')[:100]
                print(f"  Description: {desc}...")

        return True

    except Exception as e:
        print(f"\nX FAILED: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response text: {e.response.text[:500]}")
        return False


def test_github_graphql_api():
    """Test GitHub GraphQL API connectivity."""
    print("\n" + "="*60)
    print("Testing GitHub GraphQL API")
    print("="*60)

    token = os.getenv('GITHUB_TOKEN')

    if not token:
        print("X FAILED: GITHUB_TOKEN environment variable not set")
        return False

    print(f"Token present: {bool(token)}")
    print(f"Token length: {len(token)}")
    print(f"Token prefix: {token[:7]}...")

    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github.v4+json"
    }

    # Simple query to test authentication
    query = """
    query {
      viewer {
        login
      }
      rateLimit {
        limit
        remaining
        resetAt
      }
    }
    """

    try:
        print(f"\nURL: {url}")
        print("Sending authentication test query...")

        response = requests.post(
            url,
            headers=headers,
            json={"query": query},
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 401:
            print("\nX FAILED: Authentication failed (401 Unauthorized)")
            print("  The GitHub token is invalid or expired")
            print(f"  Response: {response.text}")
            return False

        response.raise_for_status()

        data = response.json()

        if 'errors' in data:
            print("\nX FAILED: GraphQL errors")
            for error in data['errors']:
                print(f"  - {error.get('message', '')}")
            return False

        viewer = data.get('data', {}).get('viewer', {})
        rate_limit = data.get('data', {}).get('rateLimit', {})

        print(f"\n+ SUCCESS")
        print(f"  Authenticated as: {viewer.get('login', 'Unknown')}")
        print(f"  Rate limit: {rate_limit.get('remaining')}/{rate_limit.get('limit')}")
        print(f"  Reset at: {rate_limit.get('resetAt')}")

        return True

    except Exception as e:
        print(f"\nX FAILED: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response text: {e.response.text[:500]}")
        return False


def test_github_security_advisories():
    """Test GitHub Security Advisories query."""
    print("\n" + "="*60)
    print("Testing GitHub Security Advisories Query")
    print("="*60)

    token = os.getenv('GITHUB_TOKEN')

    if not token:
        print("X FAILED: GITHUB_TOKEN environment variable not set")
        return False

    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github.v4+json"
    }

    query = """
    query {
      securityVulnerabilities(
        ecosystem: PIP,
        severities: [HIGH, CRITICAL],
        first: 5,
        orderBy: {field: UPDATED_AT, direction: DESC}
      ) {
        nodes {
          advisory {
            ghsaId
            summary
            severity
            publishedAt
          }
          package {
            name
          }
          vulnerableVersionRange
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
      rateLimit {
        cost
        remaining
        resetAt
      }
    }
    """

    try:
        print("Sending security advisories query...")

        response = requests.post(
            url,
            headers=headers,
            json={"query": query},
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        response.raise_for_status()

        data = response.json()

        if 'errors' in data:
            print("\nX FAILED: GraphQL errors")
            for error in data['errors']:
                print(f"  - {error.get('message', '')}")
                print(f"    Type: {error.get('type', '')}")
            return False

        vulnerabilities = data.get('data', {}).get('securityVulnerabilities', {})
        nodes = vulnerabilities.get('nodes', [])
        rate_limit = data.get('data', {}).get('rateLimit', {})

        print(f"\n+ SUCCESS")
        print(f"  Vulnerabilities returned: {len(nodes)}")
        print(f"  Has next page: {vulnerabilities.get('pageInfo', {}).get('hasNextPage')}")
        print(f"  Rate limit cost: {rate_limit.get('cost')}")
        print(f"  Rate limit remaining: {rate_limit.get('remaining')}")

        if nodes:
            print("\n  First vulnerability:")
            vuln = nodes[0]
            advisory = vuln.get('advisory', {})
            package = vuln.get('package', {})
            ghsa_id = advisory.get('ghsaId', '')
            severity = advisory.get('severity', '')
            summary = advisory.get('summary', '')[:60]
            pkg_name = package.get('name', '')
            print(f"    ID: {ghsa_id}")
            print(f"    Package: {pkg_name}")
            print(f"    Severity: {severity}")
            print(f"    Summary: {summary}...")

        return True

    except Exception as e:
        print(f"\nX FAILED: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response text: {e.response.text[:500]}")
        return False


if __name__ == '__main__':
    print("\nAPI Connectivity Test")
    print("="*60)

    # Test NVD API
    nvd_ok = test_nvd_api()

    # Test GitHub API
    github_ok = test_github_graphql_api()

    # Test GitHub Security Advisories
    advisories_ok = test_github_security_advisories()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"NVD API: {'+ OK' if nvd_ok else 'X FAILED'}")
    print(f"GitHub API Auth: {'+ OK' if github_ok else 'X FAILED'}")
    print(f"GitHub Advisories: {'+ OK' if advisories_ok else 'X FAILED'}")
    print("="*60)

    if nvd_ok and github_ok and advisories_ok:
        print("\n+ All APIs are working correctly!")
    else:
        print("\nX Some APIs failed. Please check the details above.")
