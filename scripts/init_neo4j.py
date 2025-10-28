"""Initialize Neo4j with StreamGuard schema."""

from neo4j import GraphDatabase
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class Neo4jInitializer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="streamguard"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def initialize_schema(self):
        """Create indexes and constraints."""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT file_path_unique IF NOT EXISTS
                FOR (f:File) REQUIRE f.path IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT function_id_unique IF NOT EXISTS
                FOR (fn:Function) REQUIRE fn.id IS UNIQUE
            """)

            # Create indexes
            session.run("""
                CREATE INDEX file_name_index IF NOT EXISTS
                FOR (f:File) ON (f.name)
            """)

            session.run("""
                CREATE INDEX function_name_index IF NOT EXISTS
                FOR (fn:Function) ON (fn.name)
            """)

            session.run("""
                CREATE INDEX vulnerability_type_index IF NOT EXISTS
                FOR (v:Vulnerability) ON (v.type)
            """)

            print("✅ Neo4j schema initialized")

    def create_sample_data(self):
        """Create sample graph for testing."""
        with self.driver.session() as session:
            session.run("""
                // Create sample files
                CREATE (f1:File {path: 'auth.py', name: 'auth.py'})
                CREATE (f2:File {path: 'database.py', name: 'database.py'})

                // Create functions
                CREATE (fn1:Function {
                    id: 'auth.login',
                    name: 'login',
                    file: 'auth.py',
                    start_line: 10,
                    end_line: 25
                })
                CREATE (fn2:Function {
                    id: 'database.query_user',
                    name: 'query_user',
                    file: 'database.py',
                    start_line: 5,
                    end_line: 15
                })

                // Create relationships
                CREATE (f1)-[:CONTAINS]->(fn1)
                CREATE (f2)-[:CONTAINS]->(fn2)
                CREATE (fn1)-[:CALLS]->(fn2)

                // Create vulnerability
                CREATE (v:Vulnerability {
                    id: 'vuln_001',
                    type: 'sql_injection',
                    severity: 'high',
                    confidence: 0.95,
                    line: 12
                })
                CREATE (fn2)-[:HAS_VULNERABILITY]->(v)

                // Create taint flow
                CREATE (source:TaintSource {type: 'user_input', location: 'auth.py:10'})
                CREATE (sink:TaintSink {type: 'sql_execute', location: 'database.py:12'})
                CREATE (source)-[:FLOWS_TO]->(fn1)
                CREATE (fn1)-[:FLOWS_TO]->(fn2)
                CREATE (fn2)-[:FLOWS_TO]->(sink)
            """)

            print("✅ Sample data created")

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    initializer = Neo4jInitializer()
    initializer.initialize_schema()
    initializer.create_sample_data()
    initializer.close()
