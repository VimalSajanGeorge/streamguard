"""Test Neo4j connection and verify setup."""

import os
import sys
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test connection to Neo4j database."""
    print("[*] Testing Neo4j Connection...\n")

    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'streamguard')

    print(f"URI: {uri}")
    print(f"User: {user}")
    print(f"Password: {'*' * len(password)}\n")

    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Verify connectivity
        driver.verify_connectivity()
        print("[+] Connection successful!\n")

        # Test query
        with driver.session() as session:
            # Get Neo4j version
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            record = result.single()
            print(f"Neo4j Version: {record['versions'][0]}")
            print(f"Edition: {record['edition']}\n")

            # Get database stats
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()['node_count']
            print(f"Total nodes in database: {node_count}")

            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()['rel_count']
            print(f"Total relationships: {rel_count}\n")

            # Check if schema is initialized
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            print(f"Constraints defined: {len(constraints)}")

            if len(constraints) > 0:
                print("  Existing constraints:")
                for constraint in constraints:
                    print(f"    - {constraint.get('name', 'unnamed')}")
            else:
                print("  [~] No constraints found. Run: python scripts/init_neo4j.py")

        driver.close()
        print("\n[+] Neo4j connection test passed!")
        return True

    except ServiceUnavailable:
        print("[-] Could not connect to Neo4j")
        print("\nTroubleshooting:")
        print("  1. Check if Neo4j container is running:")
        print("     docker ps | grep neo4j")
        print("  2. Start Neo4j if not running:")
        print("     docker-compose up -d")
        print("  3. Wait 30-60 seconds for Neo4j to fully start")
        print("  4. Check logs:")
        print("     docker logs streamguard-neo4j")
        return False

    except AuthError:
        print("[-] Authentication failed")
        print("\nTroubleshooting:")
        print("  1. Check credentials in .env file")
        print("  2. Default credentials: neo4j/streamguard")
        print("  3. Reset Neo4j:")
        print("     docker-compose down -v")
        print("     docker-compose up -d")
        return False

    except Exception as e:
        print(f"[-] Error: {e}")
        return False

def main():
    """Run Neo4j connection test."""
    success = test_neo4j_connection()
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
