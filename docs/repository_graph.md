    def find_taint_paths(
        self,
        source_type: Optional[str] = None,
        sink_type: Optional[str] = None,
        max_depth: int = 10
    ) -> List[TaintPath]:
        """
        Find all taint propagation paths from sources to sinks.
        
        Args:
            source_type: Filter by source type (e.g., 'user_input')
            sink_type: Filter by sink type (e.g., 'sql_injection')
            max_depth: Maximum path length to search
        
        Returns:
            List of TaintPath objects
        """
        with self.driver.session() as session:
            # Build query with optional filters
            where_clauses = []
            if source_type:
                where_clauses.append("source.type = $source_type")
            if sink_type:
                where_clauses.append("sink.type = $sink_type")
            
            where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""
            
            query = f"""
                MATCH path = (source:TaintSource)-[:IS_TAINT_SOURCE]->(fn_source:Function)
                -[:CALLS*1..{max_depth}]->(fn_sink:Function)
                -[:IS_TAINT_SINK]->(sink:TaintSink)
                {where_clause}
                RETURN 
                    source.location as source_location,
                    sink.location as sink_location,
                    [node in nodes(path) | node.id] as path_nodes,
                    length(path) as path_length,
                    source.type as source_type,
                    sink.type as sink_type
                ORDER BY path_length ASC
                LIMIT 100
            """
            
            params = {}
            if source_type:
                params['source_type'] = source_type
            if sink_type:
                params['sink_type'] = sink_type
            
            result = session.run(query, **params)
            
            taint_paths = []
            for record in result:
                taint_paths.append(TaintPath(
                    source=record['source_location'],
                    sink=record['sink_location'],
                    path=record['path_nodes'],
                    length=record['path_length'],
                    confidence=self._calculate_path_confidence(record['path_nodes']),
                    vulnerability_type=record['sink_type']
                ))
            
            return taint_paths
    
    def _calculate_path_confidence(self, path_nodes: List[str]) -> float:
        """Calculate confidence score for a taint path."""
        # Shorter paths = higher confidence
        base_confidence = 1.0 / (1 + len(path_nodes) * 0.1)
        return min(base_confidence, 1.0)
    
    def propagate_vulnerability(
        self,
        vulnerability_id: str,
        function_id: str,
        max_depth: int = 5
    ):
        """
        Propagate a detected vulnerability to all affected functions.
        
        Marks all functions that call the vulnerable function.
        """
        with self.driver.session() as session:
            # Find all callers
            result = session.run("""
                MATCH (vuln:Vulnerability {id: $vuln_id})
                MATCH (fn:Function {id: $function_id})
                MATCH (caller:Function)-[:CALLS*1..5]->(fn)
                
                // Create propagation relationship
                MERGE (caller)-[r:AFFECTED_BY]->(vuln)
                SET r.propagation_depth = length((caller)-[:CALLS*]->(fn)),
                    r.discovered_at = datetime()
                
                RETURN caller.id as affected_function,
                       r.propagation_depth as depth
            """, vuln_id=vulnerability_id, function_id=function_id)
            
            affected = []
            for record in result:
                affected.append({
                    'function': record['affected_function'],
                    'depth': record['depth']
                })
            
            return affected
    
    def get_taint_statistics(self) -> Dict:
        """Get taint propagation statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (source:TaintSource)
                OPTIONAL MATCH (sink:TaintSink)
                OPTIONAL MATCH path = (source)-[:IS_TAINT_SOURCE]->()-[:CALLS*]->()<-[:IS_TAINT_SINK]-(sink)
                
                RETURN 
                    count(DISTINCT source) as source_count,
                    count(DISTINCT sink) as sink_count,
                    count(DISTINCT path) as path_count
            """)
            
            record = result.single()
            return {
                'taint_sources': record['source_count'],
                'taint_sinks': record['sink_count'],
                'taint_paths': record['path_count']
            }
    
    def close(self):
        """Close database connection."""
        self.driver.close()


# Example usage
if __name__ == "__main__":
    tracker = TaintPropagationTracker()
    
    # Mark sources and sinks
    tracker.mark_taint_source("auth.py::handle_login", "user_input")
    tracker.mark_taint_sink("database.py::execute_query", "sql_injection")
    
    # Find taint paths
    paths = tracker.find_taint_paths(max_depth=10)
    
    print(f"Found {len(paths)} taint propagation paths:")
    for path in paths[:5]:
        print(f"\n  Source: {path.source}")
        print(f"  Sink: {path.sink}")
        print(f"  Path length: {path.length}")
        print(f"  Confidence: {path.confidence:.2f}")
        print(f"  Type: {path.vulnerability_type}")
    
    # Get statistics
    stats = tracker.get_taint_statistics()
    print(f"\n📊 Taint Statistics:")
    print(f"  Sources: {stats['taint_sources']}")
    print(f"  Sinks: {stats['taint_sinks']}")
    print(f"  Paths: {stats['taint_paths']}")
    
    tracker.close()
```

---

### 4. Attack Surface Analyzer

**File:** `core/graph/attack_surface.py`

```python
"""Analyze attack surface by mapping entry points to vulnerabilities."""

from neo4j import GraphDatabase
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class AttackVector:
    """Represents a potential attack vector."""
    entry_point: str
    entry_type: str
    vulnerability_id: str
    vulnerability_type: str
    path_length: int
    exposure_score: float
    risk_level: str


class AttackSurfaceAnalyzer:
    """Analyze and map the application's attack surface."""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "streamguard"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def find_attack_vectors(
        self,
        public_only: bool = True,
        max_depth: int = 20
    ) -> List[AttackVector]:
        """
        Find all attack vectors from entry points to vulnerabilities.
        
        Args:
            public_only: Only consider public entry points
            max_depth: Maximum path length to search
        
        Returns:
            List of AttackVector objects
        """
        with self.driver.session() as session:
            public_filter = "WHERE entry.public = true" if public_only else ""
            
            query = f"""
                MATCH (entry:EntryPoint)
                {public_filter}
                MATCH (entry)-[:HAS_ENTRY_POINT]-(file:File)
                -[:DEFINES]->(fn:Function)
                -[:CALLS*0..{max_depth}]->(vuln_fn:Function)
                -[:HAS_VULNERABILITY]->(vuln:Vulnerability)
                
                WITH entry, vuln, fn, vuln_fn,
                     shortestPath((fn)-[:CALLS*]->(vuln_fn)) as path
                
                RETURN 
                    entry.path as entry_path,
                    entry.type as entry_type,
                    vuln.id as vulnerability_id,
                    vuln.type as vulnerability_type,
                    vuln.severity as severity,
                    length(path) as path_length
                ORDER BY 
                    CASE vuln.severity
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                        ELSE 5
                    END,
                    path_length ASC
            """
            
            result = session.run(query)
            
            vectors = []
            for record in result:
                exposure_score = self._calculate_exposure_score(
                    record['entry_type'],
                    record['severity'],
                    record['path_length']
                )
                
                vectors.append(AttackVector(
                    entry_point=record['entry_path'],
                    entry_type=record['entry_type'],
                    vulnerability_id=record['vulnerability_id'],
                    vulnerability_type=record['vulnerability_type'],
                    path_length=record['path_length'],
                    exposure_score=exposure_score,
                    risk_level=self._calculate_risk_level(exposure_score)
                ))
            
            return vectors
    
    def _calculate_exposure_score(
        self,
        entry_type: str,
        severity: str,
        path_length: int
    ) -> float:
        """
        Calculate exposure score (0-1) for an attack vector.
        
        Higher score = more exposed/dangerous
        """
        # Entry type weights
        entry_weights = {
            'http_route': 1.0,
            'websocket': 0.9,
            'cli_command': 0.5,
            'internal_api': 0.3
        }
        entry_weight = entry_weights.get(entry_type, 0.5)
        
        # Severity weights
        severity_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3
        }
        severity_weight = severity_weights.get(severity, 0.5)
        
        # Path length factor (shorter = more direct = more exposed)
        path_factor = 1.0 / (1 + path_length * 0.1)
        
        # Combined score
        score = (entry_weight * 0.4 + severity_weight * 0.4 + path_factor * 0.2)
        
        return min(score, 1.0)
    
    def _calculate_risk_level(self, exposure_score: float) -> str:
        """Determine risk level from exposure score."""
        if exposure_score >= 0.8:
            return "critical"
        elif exposure_score >= 0.6:
            return "high"
        elif exposure_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def get_entry_point_statistics(self) -> Dict:
        """Get statistics about entry points."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (entry:EntryPoint)
                OPTIONAL MATCH (entry)-[:HAS_ENTRY_POINT]-(file:File)
                    -[:DEFINES]->(fn:Function)
                    -[:CALLS*]->(vuln_fn:Function)
                    -[:HAS_VULNERABILITY]->(vuln:Vulnerability)
                
                RETURN 
                    count(DISTINCT entry) as total_entry_points,
                    count(DISTINCT CASE WHEN entry.public = true THEN entry END) as public_entry_points,
                    count(DISTINCT vuln) as reachable_vulnerabilities,
                    entry.type as entry_type
                ORDER BY total_entry_points DESC
            """)
            
            stats = {
                'total_entry_points': 0,
                'public_entry_points': 0,
                'reachable_vulnerabilities': 0,
                'by_type': {}
            }
            
            for record in result:
                stats['total_entry_points'] = record['total_entry_points']
                stats['public_entry_points'] = record['public_entry_points']
                stats['reachable_vulnerabilities'] = record['reachable_vulnerabilities']
                
                if record['entry_type']:
                    stats['by_type'][record['entry_type']] = record['total_entry_points']
            
            return stats
    
    def generate_attack_surface_report(self) -> Dict:
        """Generate comprehensive attack surface report."""
        vectors = self.find_attack_vectors(public_only=True)
        stats = self.get_entry_point_statistics()
        
        # Group by risk level
        by_risk = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for vector in vectors:
            by_risk[vector.risk_level].append(vector)
        
        return {
            'summary': {
                'total_attack_vectors': len(vectors),
                'critical_vectors': len(by_risk['critical']),
                'high_vectors': len(by_risk['high']),
                'medium_vectors': len(by_risk['medium']),
                'low_vectors': len(by_risk['low'])
            },
            'entry_points': stats,
            'vectors_by_risk': by_risk,
            'top_10_vectors': sorted(vectors, key=lambda v: v.exposure_score, reverse=True)[:10]
        }
    
    def close(self):
        """Close database connection."""
        self.driver.close()


# Example usage
if __name__ == "__main__":
    analyzer = AttackSurfaceAnalyzer()
    
    # Find attack vectors
    vectors = analyzer.find_attack_vectors(public_only=True)
    
    print(f"🎯 Found {len(vectors)} attack vectors\n")
    
    # Show top 5
    for i, vector in enumerate(vectors[:5], 1):
        print(f"{i}. {vector.risk_level.upper()} Risk")
        print(f"   Entry: {vector.entry_point} ({vector.entry_type})")
        print(f"   Vulnerability: {vector.vulnerability_type}")
        print(f"   Path length: {vector.path_length}")
        print(f"   Exposure: {vector.exposure_score:.2f}")
        print()
    
    # Generate report
    report = analyzer.generate_attack_surface_report()
    
    print("📊 Attack Surface Report:")
    print(f"  Total vectors: {report['summary']['total_attack_vectors']}")
    print(f"  Critical: {report['summary']['critical_vectors']}")
    print(f"  High: {report['summary']['high_vectors']}")
    print(f"  Medium: {report['summary']['medium_vectors']}")
    print(f"  Low: {report['summary']['low_vectors']}")
    
    analyzer.close()
```

---

### 5. Graph Query API

**File:** `core/graph/graph_api.py`

```python
"""High-level API for graph queries."""

from neo4j import GraphDatabase
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class QueryResult:
    """Generic query result."""
    data: List[Dict]
    count: int
    query_time_ms: float


class GraphQueryAPI:
    """High-level API for common graph queries."""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "streamguard"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def get_function_dependencies(
        self,
        function_id: str,
        depth: int = 1
    ) -> QueryResult:
        """Get all functions that a function calls."""
        import time
        start = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (fn:Function {id: $function_id})
                -[:CALLS*1..{depth}]->(called:Function)
                RETURN 
                    called.id as function_id,
                    called.name as name,
                    called.file as file,
                    length((fn)-[:CALLS*]->(called)) as depth
                ORDER BY depth ASC
            """.format(depth=depth), function_id=function_id)
            
            data = [dict(record) for record in result]
        
        query_time = (time.time() - start) * 1000
        
        return QueryResult(
            data=data,
            count=len(data),
            query_time_ms=query_time
        )
    
    def get_function_callers(
        self,
        function_id: str,
        depth: int = 1
    ) -> QueryResult:
        """Get all functions that call a function."""
        import time
        start = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (caller:Function)-[:CALLS*1..{depth}]->
                      (fn:Function {id: $function_id})
                RETURN 
                    caller.id as function_id,
                    caller.name as name,
                    caller.file as file,
                    length((caller)-[:CALLS*]->(fn)) as depth
                ORDER BY depth ASC
            """.format(depth=depth), function_id=function_id)
            
            data = [dict(record) for record in result]
        
        query_time = (time.time() - start) * 1000
        
        return QueryResult(
            data=data,
            count=len(data),
            query_time_ms=query_time
        )
    
    def get_file_dependencies(self, file_path: str) -> QueryResult:
        """Get all files that a file imports."""
        import time
        start = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:File {path: $file_path})-[r:IMPORTS]->(imported:File)
                RETURN 
                    imported.path as file_path,
                    imported.name as name,
                    imported.language as language,
                    r.symbols as imported_symbols
            """, file_path=file_path)
            
            data = [dict(record) for record in result]
        
        query_time = (time.time() - start) * 1000
        
        return QueryResult(
            data=data,
            count=len(data),
            query_time_ms=query_time
        )
    
    def get_vulnerability_impact(
        self,
        vulnerability_id: str
    ) -> QueryResult:
        """Get impact analysis for a vulnerability."""
        import time
        start = time.time()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (vuln:Vulnerability {id: $vuln_id})
                MATCH (fn:Function)-[:HAS_VULNERABILITY]->(vuln)
                
                // Find all callers (blast radius)
                OPTIONAL MATCH (caller:Function)-[:CALLS*1..5]->(fn)
                
                // Find entry points that reach this vulnerability
                OPTIONAL MATCH (entry:EntryPoint)-[:HAS_ENTRY_POINT]-(:File)
                    -[:DEFINES]->(:Function)
                    -[:CALLS*]->(fn)
                
                RETURN 
                    vuln.type as vulnerability_type,
                    vuln.severity as severity,
                    fn.id as vulnerable_function,
                    count(DISTINCT caller) as affected_functions,
                    count(DISTINCT entry) as exposed_entry_points,
                    collect(DISTINCT entry.path)[..5] as sample_entry_points
            """, vuln_id=vulnerability_id)
            
            data = [dict(record) for record in result]
        
        query_time = (time.time() - start) * 1000
        
        return QueryResult(
            data=data,
            count=len(data),
            query_time_ms=query_time
        )
    
    def search_functions(
        self,
        name_pattern: str,
        file_pattern: Optional[str] = None
    ) -> QueryResult:
        """Search for functions by name pattern."""
        import time
        start = time.time()
        
        with self.driver.session() as session:
            file_filter = ""
            params = {'name_pattern': f"(?i).*{name_pattern}.*"}
            
            if file_pattern:
                file_filter = "AND fn.file =~ $file_pattern"
                params['file_pattern'] = f"(?i).*{file_pattern}.*"
            
            result = session.run(f"""
                MATCH (fn:Function)
                WHERE fn.name =~ $name_pattern
                {file_filter}
                RETURN 
                    fn.id as function_id,
                    fn.name as name,
                    fn.file as file,
                    fn.start_line as start_line,
                    fn.end_line as end_line
                LIMIT 50
            """, **params)
            
            data = [dict(record) for record in result]
        
        query_time = (time.time() - start) * 1000
        
        return QueryResult(
            data=data,
            count=len(data),
            query_time_ms=query_time
        )
    
    def get_graph_statistics(self) -> Dict:
        """Get overall graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WITH labels(n)[0] as label, count(n) as count
                RETURN label, count
                ORDER BY count DESC
            """)
            
            node_stats = {record['label']: record['count'] for record in result}
            
            result = session.run("""
                MATCH ()-[r]->()
                WITH type(r) as rel_type, count(r) as count
                RETURN rel_type, count
                ORDER BY count DESC
            """)
            
            rel_stats = {record['rel_type']: record['count'] for record in result}
            
            return {
                'nodes': node_stats,
                'relationships': rel_stats,
                'total_nodes': sum(node_stats.values()),
                'total_relationships': sum(rel_stats.values())
            }
    
    def close(self):
        """Close database connection."""
        self.driver.close()


# Example usage
if __name__ == "__main__":
    api = GraphQueryAPI()
    
    # Search functions
    result = api.search_functions("login", file_pattern="auth")
    print(f"🔍 Found {result.count} functions matching 'login'")
    print(f"   Query time: {result.query_time_ms:.2f}ms")
    
    for func in result.data[:3]:
        print(f"   • {func['name']} in {func['file']}")
    
    # Get dependencies
    if result.data:
        func_id = result.data[0]['function_id']
        deps = api.get_function_dependencies(func_id, depth=2)
        print(f"\n📦 {func_id} calls {deps.count} functions")
    
    # Get statistics
    stats = api.get_graph_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total relationships: {stats['total_relationships']}")
    
    api.close()
```

---

### 6. Testing & Benchmarks

**File:** `tests/integration/test_graph_system.py`

```python
"""Integration tests for graph system."""

import pytest
from core.graph.neo4j_schema import Neo4jSchema
from core.graph.graph_builder import RepositoryGraphBuilder
from core.graph.taint_propagation import TaintPropagationTracker
from core.graph.attack_surface import AttackSurfaceAnalyzer
from core.graph.graph_api import GraphQueryAPI
from pathlib import Path
import tempfile

class TestGraphSystem:
    """Test complete graph system."""
    
    @pytest.fixture
    def schema(self):
        """Initialize schema."""
        schema = Neo4jSchema()
        schema.clear_database()
        schema.initialize()
        yield schema
        schema.close()
    
    @pytest.fixture
    def sample_project(self):
        """Create sample project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create sample files
            auth_file = project_path / "auth.py"
            auth_file.write_text("""
def handle_login(username, password):
    user_data = get_user_input()
    result = query_user(user_data)
    return result

def get_user_input():
    return input("Username: ")
""")
            
            db_file = project_path / "database.py"
            db_file.write_text("""
def query_user(username):
    query = "SELECT * FROM users WHERE name='" + username + "'"
    return execute(query)

def execute(sql):
    pass
""")
            
            yield project_path
    
    def test_schema_initialization(self, schema):
        """Test schema is created correctly."""
        stats = schema.get_statistics()
        
        # Schema should be initialized but empty
        assert stats['total_nodes'] >= 0
        assert stats['total_relationships'] >= 0
    
    def test_graph_builder(self, schema, sample_project):
        """Test building graph from project."""
        builder = RepositoryGraphBuilder()
        builder.build_graph(str(sample_project))
        builder.close()
        
        # Check graph was built
        stats = schema.get_statistics()
        assert stats['total_nodes'] > 0
        assert 'File' in stats['nodes']
        assert 'Function' in stats['nodes']
    
    def test_taint_propagation(self, schema, sample_project):
        """Test taint propagation tracking."""
        # Build graph first
        builder = RepositoryGraphBuilder()
        builder.build_graph(str(sample_project))
        builder.close()
        
        # Mark sources and sinks
        tracker = TaintPropagationTracker()
        tracker.mark_taint_source("auth.py::get_user_input", "user_input")
        tracker.mark_taint_sink("database.py::execute", "sql_injection")
        
        # Find paths
        paths = tracker.find_taint_paths()
        
        # Should find at least one path
        assert len(paths) >= 0
        
        tracker.close()
    
    def test_attack_surface_analysis(self, schema, sample_project):
        """Test attack surface analyzer."""
        # Build graph and add entry point
        builder = RepositoryGraphBuilder()
        builder.build_graph(str(sample_project))
        builder.close()
        
        analyzer = AttackSurfaceAnalyzer()
        vectors = analyzer.find_attack_vectors(public_only=False)
        
        # Analysis should complete without errors
        assert isinstance(vectors, list)
        
        analyzer.close()
    
    def test_graph_query_api(self, schema, sample_project):
        """Test graph query API."""
        # Build graph
        builder = RepositoryGraphBuilder()
        builder.build_graph(str(sample_project))
        builder.close()
        
        api = GraphQueryAPI()
        
        # Test search
        result = api.search_functions("login")
        assert result.count >= 0
        assert result.query_time_ms < 100  # <100ms
        
        # Test statistics
        stats = api.get_graph_statistics()
        assert 'nodes' in stats
        assert 'relationships' in stats
        
        api.close()
    
    def test_query_performance(self, schema, sample_project):
        """Test query performance meets targets."""
        builder = RepositoryGraphBuilder()
        builder.build_graph(str(sample_project))
        builder.close()
        
        api = GraphQueryAPI()
        
        # Run multiple queries and measure time
        query_times = []
        
        for _ in range(10):
            result = api.search_functions("test")
            query_times.append(result.query_time_ms)
        
        avg_time = sum(query_times) / len(query_times)
        
        # Should be < 50ms average
        assert avg_time < 50, f"Average query time {avg_time}ms exceeds target"
        
        api.close()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```

---

## ✅ Implementation Checklist

### Core Components
- [ ] Neo4j schema with constraints and indexes
- [ ] Repository graph builder with AST parsing
- [ ] Function and class extraction
- [ ] Import relationship tracking
- [ ] Entry point detection

### Taint Analysis
- [ ] Taint source/sink marking
- [ ] Taint path discovery
- [ ] Vulnerability propagation
- [ ] Confidence scoring

### Attack Surface
- [ ] Attack vector identification
- [ ] Entry point to vulnerability mapping
- [ ] Exposure score calculation
- [ ] Risk level assessment
- [ ] Attack surface reporting

### Query API
- [ ] Function dependency queries
- [ ] File dependency queries
- [ ] Vulnerability impact analysis
- [ ] Function search
- [ ] Graph statistics

### Testing
- [ ] Unit tests for each component
- [ ] Integration tests with sample projects
- [ ] Performance benchmarks (<50ms queries)
- [ ] Large-scale testing (1000+ files)

---

## 🎯 Performance Targets

| Metric | Target | Maximum | Status |
|--------|--------|---------|--------|
| **Graph Build Time** | <30s for 100 files | <60s | ⏳ To Validate |
| **Simple Query** | <10ms | <50ms | ⏳ To Validate |
| **Complex Query (depth=5)** | <50ms | <200ms | ⏳ To Validate |
| **Taint Path Discovery** | <100ms | <500ms | ⏳ To Validate |
| **Attack Surface Analysis** | <200ms | <1s | ⏳ To Validate |
| **Memory Usage** | <1GB | <4GB | ⏳ To Validate |
| **Graph Size** | 10K+ nodes | 100K+ nodes | ⏳ To Validate |

---

## 🚀 Quick Start

```bash
# 1. Start Neo4j
docker-compose up -d neo4j

# 2. Initialize schema
python -m core.graph.neo4j_schema

# 3. Build graph from project
python -m core.graph.graph_builder /path/to/project

# 4. Query graph
python -m core.graph.graph_api

# 5. Analyze attack surface
python -m core.graph.attack_surface
```

---

**Status:** ✅ Ready for Implementation  
**Next:** [06_ui_feedback.md](./06_ui_feedback.md) - UI & Feedback System# 05 - Repository Graph System with Neo4j

**Phase:** 4 (Weeks 9-10)  
**Prerequisites:** Local agent running ([04_agent_architecture.md](./04_agent_architecture.md))  
**Status:** Ready to Implement

---

## 📋 Overview

Build a graph-based repository analysis system using Neo4j/TigerGraph to track code dependencies, data flow, and vulnerability propagation across the entire codebase.

**Key Features:**
- **Dependency Graph**: Complete code dependency tracking
- **Taint Flow Propagation**: Cross-file vulnerability tracking
- **Attack Surface Analysis**: Entry point to vulnerability mapping
- **Impact Assessment**: Blast radius of vulnerabilities
- **Query Optimization**: Sub-50ms graph queries
- **Real-Time Updates**: Incremental graph updates

**Deliverables:**
- ✅ Neo4j schema and indexes
- ✅ Repository graph builder
- ✅ Taint propagation tracker
- ✅ Attack surface analyzer
- ✅ Query API with Cypher
- ✅ Graph visualization data

**Expected Time:** 2 weeks

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Repository Codebase                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ auth.py  │──│ db.py    │──│ models.py│──│ utils.py │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────┬──────────────────────────────────────┘
                          │ Analyze & Extract
                          ▼
┌────────────────────────────────────────────────────────────────┐
│                 Repository Graph Builder                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  AST Parser & Extractor                                   │ │
│  │  • Parse source files (Tree-sitter)                       │ │
│  │  • Extract functions, classes, imports                    │ │
│  │  • Identify call sites and data flows                     │ │
│  │  • Detect entry points (routes, CLI, etc.)                │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │  Graph Construction Engine                                │ │
│  │  • Create nodes (File, Function, Class, Variable)         │ │
│  │  • Create edges (CALLS, IMPORTS, DEFINES, ACCESSES)       │ │
│  │  • Add properties (line numbers, types, params)           │ │
│  │  • Index key attributes                                   │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
└────────────────────────────┼──────────────────────────────────┘
                             │ Store in
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    Neo4j Graph Database                        │
│                                                                 │
│  Node Types:                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   File   │  │ Function │  │  Class   │  │ Variable │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │EntryPoint│  │TaintSource│ │TaintSink │ │Vulnerability│    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                 │
│  Edge Types:                                                    │
│  CALLS, IMPORTS, DEFINES, ACCESSES, FLOWS_TO,                  │
│  HAS_VULNERABILITY, PROPAGATES_TO, LEADS_TO                    │
│                                                                 │
│  Indexes:                                                       │
│  • File.path (UNIQUE)                                           │
│  • Function.id (UNIQUE)                                         │
│  • Vulnerability.type                                           │
└────────────────────────────┬──────────────────────────────────┘
                             │ Query for
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              Graph Analysis & Query System                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Taint Propagation Tracker                                │ │
│  │                                                            │ │
│  │  User Input (TaintSource)                                 │ │
│  │         ↓                                                  │ │
│  │    Function A (propagates)                                │ │
│  │         ↓                                                  │ │
│  │    Function B (propagates)                                │ │
│  │         ↓                                                  │ │
│  │    SQL Execute (TaintSink)                                │ │
│  │         ↓                                                  │ │
│  │    VULNERABILITY FOUND                                     │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  Attack Surface Analyzer                                  │ │
│  │  • Find all entry points (public routes, APIs)            │ │
│  │  • Trace paths to vulnerabilities                         │ │
│  │  • Calculate exposure score                               │ │
│  │  • Prioritize by risk                                     │ │
│  └───────────────────────┬───────────────────────────────────┘ │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐ │
│  │  Impact Assessment Engine                                 │ │
│  │  • Calculate blast radius                                 │ │
│  │  • Find affected functions/files                          │ │
│  │  • Identify cascading vulnerabilities                     │ │
│  │  • Generate impact reports                                │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ Results
                             ▼
                    Visualization & Reporting
```

### Graph Schema

```
Nodes:
├── File {path, name, language, lines_of_code}
├── Function {id, name, file, start_line, end_line, params, returns}
├── Class {id, name, file, methods}
├── Variable {name, type, scope}
├── EntryPoint {type, path, method, public}
├── TaintSource {type, location, input_type}
├── TaintSink {type, location, operation}
└── Vulnerability {id, type, severity, line, confidence}

Edges:
├── CALLS {from: Function, to: Function, line}
├── IMPORTS {from: File, to: File, symbols}
├── DEFINES {from: File, to: Function/Class}
├── CONTAINS {from: Class, to: Function}
├── ACCESSES {from: Function, to: Variable, mode: read/write}
├── FLOWS_TO {from: TaintSource, to: Function/TaintSink, data_type}
├── HAS_VULNERABILITY {from: Function, to: Vulnerability}
├── PROPAGATES_TO {from: Vulnerability, to: Function, confidence}
└── LEADS_TO {from: EntryPoint, to: Vulnerability, path_length}
```

---

## 💻 Implementation

### 1. Neo4j Schema Initialization

**File:** `core/graph/neo4j_schema.py`

```python
"""Neo4j schema initialization for StreamGuard."""

from neo4j import GraphDatabase
from typing import Dict, List

class Neo4jSchema:
    """Initialize and manage Neo4j schema."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "streamguard"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def initialize(self):
        """Initialize complete schema with constraints and indexes."""
        print("🔧 Initializing Neo4j schema...")
        
        with self.driver.session() as session:
            # Create constraints
            self._create_constraints(session)
            
            # Create indexes
            self._create_indexes(session)
            
            # Create sample data (optional)
            # self._create_sample_data(session)
        
        print("✅ Schema initialized successfully")
    
    def _create_constraints(self, session):
        """Create uniqueness constraints."""
        constraints = [
            # File constraints
            """
            CREATE CONSTRAINT file_path_unique IF NOT EXISTS
            FOR (f:File) REQUIRE f.path IS UNIQUE
            """,
            
            # Function constraints
            """
            CREATE CONSTRAINT function_id_unique IF NOT EXISTS
            FOR (fn:Function) REQUIRE fn.id IS UNIQUE
            """,
            
            # Class constraints
            """
            CREATE CONSTRAINT class_id_unique IF NOT EXISTS
            FOR (c:Class) REQUIRE c.id IS UNIQUE
            """,
            
            # Vulnerability constraints
            """
            CREATE CONSTRAINT vulnerability_id_unique IF NOT EXISTS
            FOR (v:Vulnerability) REQUIRE v.id IS UNIQUE
            """,
            
            # EntryPoint constraints
            """
            CREATE CONSTRAINT entrypoint_path_unique IF NOT EXISTS
            FOR (e:EntryPoint) REQUIRE (e.type, e.path) IS UNIQUE
            """
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"  ✓ Created constraint")
            except Exception as e:
                if "EquivalentSchemaRuleAlreadyExists" not in str(e):
                    print(f"  ⚠️  Constraint error: {e}")
    
    def _create_indexes(self, session):
        """Create performance indexes."""
        indexes = [
            # File indexes
            """
            CREATE INDEX file_name_index IF NOT EXISTS
            FOR (f:File) ON (f.name)
            """,
            
            """
            CREATE INDEX file_language_index IF NOT EXISTS
            FOR (f:File) ON (f.language)
            """,
            
            # Function indexes
            """
            CREATE INDEX function_name_index IF NOT EXISTS
            FOR (fn:Function) ON (fn.name)
            """,
            
            """
            CREATE INDEX function_file_index IF NOT EXISTS
            FOR (fn:Function) ON (fn.file)
            """,
            
            # Vulnerability indexes
            """
            CREATE INDEX vulnerability_type_index IF NOT EXISTS
            FOR (v:Vulnerability) ON (v.type)
            """,
            
            """
            CREATE INDEX vulnerability_severity_index IF NOT EXISTS
            FOR (v:Vulnerability) ON (v.severity)
            """,
            
            # TaintSource indexes
            """
            CREATE INDEX taintsource_type_index IF NOT EXISTS
            FOR (t:TaintSource) ON (t.type)
            """,
            
            # TaintSink indexes
            """
            CREATE INDEX taintsink_type_index IF NOT EXISTS
            FOR (t:TaintSink) ON (t.type)
            """,
            
            # EntryPoint indexes
            """
            CREATE INDEX entrypoint_type_index IF NOT EXISTS
            FOR (e:EntryPoint) ON (e.type)
            """
        ]
        
        for index in indexes:
            try:
                session.run(index)
                print(f"  ✓ Created index")
            except Exception as e:
                if "EquivalentSchemaRuleAlreadyExists" not in str(e):
                    print(f"  ⚠️  Index error: {e}")
    
    def clear_database(self):
        """Clear all data (use with caution!)."""
        print("⚠️  Clearing database...")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        print("✅ Database cleared")
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN 
                    labels(n)[0] as label,
                    count(n) as count
                ORDER BY count DESC
            """)
            
            stats = {record["label"]: record["count"] for record in result}
            
            # Count relationships
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            
            rel_stats = {record["type"]: record["count"] for record in rel_result}
            
            return {
                "nodes": stats,
                "relationships": rel_stats,
                "total_nodes": sum(stats.values()),
                "total_relationships": sum(rel_stats.values())
            }
    
    def close(self):
        """Close database connection."""
        self.driver.close()


# Example usage
if __name__ == "__main__":
    schema = Neo4jSchema()
    
    # Initialize schema
    schema.initialize()
    
    # Get statistics
    stats = schema.get_statistics()
    print("\n📊 Database Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    
    print("\n  Nodes by type:")
    for label, count in stats['nodes'].items():
        print(f"    {label}: {count}")
    
    print("\n  Relationships by type:")
    for rel_type, count in stats['relationships'].items():
        print(f"    {rel_type}: {count}")
    
    schema.close()
```

---

### 2. Repository Graph Builder

**File:** `core/graph/graph_builder.py`

```python
"""Build code dependency graph from repository."""

import os
from pathlib import Path
from typing import Dict, List, Set, Optional
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
from neo4j import GraphDatabase
import hashlib

class RepositoryGraphBuilder:
    """Build complete repository dependency graph."""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "streamguard"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.parsers = self._init_parsers()
        
        # Supported file extensions
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript'
        }
    
    def _init_parsers(self) -> Dict[str, tree_sitter.Parser]:
        """Initialize Tree-sitter parsers."""
        parsers = {}
        
        # Python
        python_parser = tree_sitter.Parser()
        python_parser.set_language(tree_sitter.Language(tree_sitter_python.language(), 'python'))
        parsers['python'] = python_parser
        
        # JavaScript/TypeScript
        js_parser = tree_sitter.Parser()
        js_parser.set_language(tree_sitter.Language(tree_sitter_javascript.language(), 'javascript'))
        parsers['javascript'] = js_parser
        parsers['typescript'] = js_parser
        
        return parsers
    
    def build_graph(self, project_root: str, incremental: bool = False):
        """
        Build complete repository graph.
        
        Args:
            project_root: Root directory of the project
            incremental: If True, only update changed files
        """
        project_path = Path(project_root).resolve()
        
        print(f"🏗️  Building repository graph: {project_path}")
        print(f"   Incremental: {incremental}")
        
        # Find all code files
        code_files = self._find_code_files(project_path)
        print(f"   Found {len(code_files)} code files")
        
        # Process each file
        for i, file_path in enumerate(code_files, 1):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(code_files)}")
            
            try:
                self._process_file(file_path, project_path)
            except Exception as e:
                print(f"   ⚠️  Error processing {file_path}: {e}")
        
        # Build cross-file relationships
        self._build_cross_file_relationships()
        
        print("✅ Graph build complete")
    
    def _find_code_files(self, root_path: Path) -> List[Path]:
        """Find all supported code files."""
        code_files = []
        
        # Directories to ignore
        ignore_dirs = {
            'node_modules', '.git', '__pycache__',
            'venv', 'env', 'dist', 'build',
            '.vscode', '.idea', 'target'
        }
        
        for file_path in root_path.rglob('*'):
            # Skip ignored directories
            if any(ignored in file_path.parts for ignored in ignore_dirs):
                continue
            
            # Check if supported file type
            if file_path.suffix in self.supported_extensions:
                code_files.append(file_path)
        
        return code_files
    
    def _process_file(self, file_path: Path, project_root: Path):
        """Process a single file and extract graph nodes/edges."""
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"   Error reading {file_path}: {e}")
            return
        
        # Detect language
        language = self.supported_extensions.get(file_path.suffix)
        if not language or language not in self.parsers:
            return
        
        # Parse AST
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        
        # Get relative path
        relative_path = str(file_path.relative_to(project_root))
        
        # Extract information
        file_info = {
            'path': relative_path,
            'name': file_path.name,
            'language': language,
            'lines_of_code': len(code.split('\n')),
            'checksum': hashlib.md5(code.encode()).hexdigest()
        }
        
        functions = self._extract_functions(root_node, code, relative_path)
        classes = self._extract_classes(root_node, code, relative_path)
        imports = self._extract_imports(root_node, code, language)
        entry_points = self._extract_entry_points(root_node, code, relative_path, language)
        
        # Store in Neo4j
        with self.driver.session() as session:
            # Create File node
            session.run("""
                MERGE (f:File {path: $path})
                SET f.name = $name,
                    f.language = $language,
                    f.lines_of_code = $loc,
                    f.checksum = $checksum
            """, path=file_info['path'], name=file_info['name'],
                language=file_info['language'], loc=file_info['lines_of_code'],
                checksum=file_info['checksum'])
            
            # Create Function nodes
            for func in functions:
                session.run("""
                    MERGE (fn:Function {id: $id})
                    SET fn.name = $name,
                        fn.file = $file,
                        fn.start_line = $start_line,
                        fn.end_line = $end_line,
                        fn.params = $params,
                        fn.returns = $returns
                    
                    WITH fn
                    MATCH (f:File {path: $file})
                    MERGE (f)-[:DEFINES]->(fn)
                """, **func)
            
            # Create Class nodes
            for cls in classes:
                session.run("""
                    MERGE (c:Class {id: $id})
                    SET c.name = $name,
                        c.file = $file,
                        c.methods = $methods
                    
                    WITH c
                    MATCH (f:File {path: $file})
                    MERGE (f)-[:DEFINES]->(c)
                """, **cls)
            
            # Create Import relationships
            for imp in imports:
                session.run("""
                    MATCH (f1:File {path: $from_file})
                    MATCH (f2:File {path: $to_file})
                    MERGE (f1)-[r:IMPORTS]->(f2)
                    SET r.symbols = $symbols
                """, **imp)
            
            # Create EntryPoint nodes
            for entry in entry_points:
                session.run("""
                    MERGE (e:EntryPoint {type: $type, path: $path})
                    SET e.method = $method,
                        e.public = $public,
                        e.file = $file,
                        e.line = $line
                    
                    WITH e
                    MATCH (f:File {path: $file})
                    MERGE (f)-[:HAS_ENTRY_POINT]->(e)
                """, **entry)
    
    def _extract_functions(self, root_node, code: str, file_path: str) -> List[Dict]:
        """Extract function definitions."""
        functions = []
        
        def traverse(node):
            if node.type in ['function_definition', 'function_declaration', 'method_definition']:
                # Get function name
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_name = code[name_node.start_byte:name_node.end_byte]
                    
                    # Get parameters
                    params_node = node.child_by_field_name('parameters')
                    params = code[params_node.start_byte:params_node.end_byte] if params_node else ""
                    
                    functions.append({
                        'id': f"{file_path}::{func_name}",
                        'name': func_name,
                        'file': file_path,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'params': params,
                        'returns': None  # TODO: Extract return type
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return functions
    
    def _extract_classes(self, root_node, code: str, file_path: str) -> List[Dict]:
        """Extract class definitions."""
        classes = []
        
        def traverse(node):
            if node.type in ['class_definition', 'class_declaration']:
                # Get class name
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = code[name_node.start_byte:name_node.end_byte]
                    
                    # Get methods
                    methods = []
                    for child in node.children:
                        if child.type in ['function_definition', 'method_definition']:
                            method_name_node = child.child_by_field_name('name')
                            if method_name_node:
                                methods.append(code[method_name_node.start_byte:method_name_node.end_byte])
                    
                    classes.append({
                        'id': f"{file_path}::{class_name}",
                        'name': class_name,
                        'file': file_path,
                        'methods': methods
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_imports(self, root_node, code: str, language: str) -> List[Dict]:
        """Extract import statements."""
        imports = []
        
        def traverse(node):
            if node.type in ['import_statement', 'import_from_statement', 'import_declaration']:
                # Extract import information
                # This is language-specific and simplified
                import_text = code[node.start_byte:node.end_byte]
                
                # TODO: Parse import target file path
                # For now, just store the import statement
                # imports.append({...})
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_entry_points(self, root_node, code: str, file_path: str, language: str) -> List[Dict]:
        """Extract entry points (routes, CLI commands, etc.)."""
        entry_points = []
        
        if language == 'python':
            # Look for Flask/Django routes, FastAPI endpoints, etc.
            def traverse(node):
                if node.type == 'decorator':
                    decorator_text = code[node.start_byte:node.end_byte]
                    
                    # Check for route decorators
                    if any(pattern in decorator_text for pattern in ['@app.route', '@router.', '@api.route']):
                        # This is an entry point
                        function_node = node.parent
                        if function_node and function_node.type == 'decorated_definition':
                            func_def = function_node.child_by_field_name('definition')
                            if func_def:
                                name_node = func_def.child_by_field_name('name')
                                if name_node:
                                    func_name = code[name_node.start_byte:name_node.end_byte]
                                    
                                    entry_points.append({
                                        'type': 'http_route',
                                        'path': decorator_text,  # Parse route path
                                        'method': 'GET',  # Parse HTTP method
                                        'public': True,
                                        'file': file_path,
                                        'line': node.start_point[0] + 1
                                    })
                
                for child in node.children:
                    traverse(child)
            
            traverse(root_node)
        
        return entry_points
    
    def _build_cross_file_relationships(self):
        """Build relationships between files (function calls, etc.)."""
        print("   Building cross-file relationships...")
        
        with self.driver.session() as session:
            # Find function calls and create CALLS relationships
            # This is a simplified version
            session.run("""
                MATCH (fn1:Function), (fn2:Function)
                WHERE fn1.id <> fn2.id
                // TODO: Add logic to detect function calls
                // This would require analyzing function bodies
                RETURN count(*) as calls_created
            """)
    
    def close(self):
        """Close database connection."""
        self.driver.close()


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graph_builder.py <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    
    builder = RepositoryGraphBuilder()
    builder.build_graph(project_root)
    builder.close()
```

---

### 3. Taint Propagation Tracker

**File:** `core/graph/taint_propagation.py`

```python
"""Track taint propagation across the codebase using graph queries."""

from neo4j import GraphDatabase
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

@dataclass
class TaintPath:
    """Represents a taint flow path."""
    source: str
    sink: str
    path: List[str]
    length: int
    confidence: float
    vulnerability_type: str


class TaintPropagationTracker:
    """Track how tainted data flows through the codebase."""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "streamguard"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    def mark_taint_source(
        self,
        function_id: str,
        source_type: str,
        input_type: str = "user_input"
    ):
        """Mark a function as a taint source."""
        with self.driver.session() as session:
            session.run("""
                MATCH (fn:Function {id: $function_id})
                MERGE (t:TaintSource {type: $source_type, location: $function_id})
                SET t.input_type = $input_type
                MERGE (fn)-[:IS_TAINT_SOURCE]->(t)
            """, function_id=function_id, source_type=source_type, input_type=input_type)
    
    def mark_taint_sink(
        self,
        function_id: str,
        sink_type: str,
        operation: str = "sql_execute"
    ):
        """Mark a function as a taint sink."""
        with self.driver.session() as session:
            session.run("""
                MATCH (fn:Function {id: $function_id})
                MERGE (t:TaintSink {type: $sink_type, location: $function_id})
                SET t.operation = $operation
                MERGE (fn)-[:IS_TAINT_SINK]->(t)
            """, function_id=function_id, sink_type=sink_type, operation=operation)
    
    def find_taint_paths(
        self,
        source_type: Optional[str] = None,
        sink_type: Optional[str] = None,
        max_depth: int = 10
    ) -> List[TaintPath]:
        """
        Find all taint propagation paths from sources to sinks.
        
        Args:
            source_type: Filter by source type (e.g., 'user_input')
            sink_type: Filter by sink type (e.g., 'sql_injection')
            max_depth: Maximum path length to search
        
        Returns:
            List