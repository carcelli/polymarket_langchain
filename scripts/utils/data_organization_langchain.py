#!/usr/bin/env python3
"""
Data Organization for LangChain Integration

This module provides utilities to organize and structure your Polymarket ML data
for optimal use with LangChain agents and tools.

Current Data Inventory:
- markets.db: 30MB, 20,716 markets (main Polymarket data)
- standalone_ml.db: 28KB, ML experiments/models/workflows
- JSON reports: Individual workflow summaries
- Memory databases: Agent conversation memory

LangChain Integration Strategy:
1. Vector Stores: Semantic search over market data and ML results
2. Document Stores: Structured retrieval of experiments and models
3. Metadata Indexing: Rich metadata for filtering and retrieval
4. Query Optimization: Fast lookups for agent interactions
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# LangChain imports for data organization
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PolymarketDataOrganizer:
    """
    Organizes Polymarket ML data for LangChain agent consumption.

    Provides:
    - Vector stores for semantic search
    - Structured document stores
    - Metadata indexing
    - Query optimization
    """

    def __init__(self, data_dir: str = "./data", vector_store_dir: str = "./vector_stores"):
        self.data_dir = Path(data_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(exist_ok=True)

        # Initialize embeddings only when needed (requires OPENAI_API_KEY)
        self._embeddings = None

    @property
    def embeddings(self):
        """Lazy initialization of embeddings."""
        if self._embeddings is None:
            try:
                self._embeddings = OpenAIEmbeddings()
            except Exception as e:
                raise RuntimeError(f"OpenAI embeddings require OPENAI_API_KEY: {e}")
        return self._embeddings

    def get_data_inventory(self) -> Dict[str, Any]:
        """
        Get comprehensive inventory of all data being created.

        Returns:
            Dictionary with data statistics and structure
        """
        inventory = {
            'timestamp': datetime.now().isoformat(),
            'total_size_mb': 0,
            'databases': {},
            'json_files': {},
            'vector_stores': {},
            'data_breakdown': {}
        }

        # Check databases
        db_files = list(self.data_dir.glob("*.db"))
        for db_file in db_files:
            size_mb = db_file.stat().st_size / (1024 * 1024)

            db_info = {
                'size_mb': round(size_mb, 2),
                'tables': {},
                'record_counts': {}
            }

            try:
                with sqlite3.connect(str(db_file)) as conn:
                    # Get table names
                    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                    db_info['tables'] = [t[0] for t in tables]

                    # Get record counts
                    for table in db_info['tables']:
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        db_info['record_counts'][table] = count

            except Exception as e:
                db_info['error'] = str(e)

            inventory['databases'][db_file.name] = db_info
            inventory['total_size_mb'] += size_mb

        # Check JSON files
        json_files = list(self.data_dir.glob("*.json")) + list(Path(".").glob("workflow_report_*.json"))
        for json_file in json_files:
            size_kb = json_file.stat().st_size / 1024

            inventory['json_files'][json_file.name] = {
                'size_kb': round(size_kb, 2),
                'path': str(json_file)
            }
            inventory['total_size_mb'] += size_kb / 1024

        # Check vector stores
        if self.vector_store_dir.exists():
            vector_files = list(self.vector_store_dir.glob("*"))
            for vf in vector_files:
                size_mb = vf.stat().st_size / (1024 * 1024)
                inventory['vector_stores'][vf.name] = round(size_mb, 2)
                inventory['total_size_mb'] += size_mb

        # Data breakdown analysis
        inventory['data_breakdown'] = self._analyze_data_breakdown()

        return inventory

    def _analyze_data_breakdown(self) -> Dict[str, Any]:
        """Analyze breakdown of different data types."""
        breakdown = {
            'markets_data': {'count': 0, 'description': 'Polymarket trading data'},
            'ml_experiments': {'count': 0, 'description': 'ML model training experiments'},
            'ml_models': {'count': 0, 'description': 'Trained ML models'},
            'workflows': {'count': 0, 'description': 'Automated ML workflow executions'},
            'reports': {'count': 0, 'description': 'Generated analysis reports'},
            'memory': {'count': 0, 'description': 'Agent conversation memory'}
        }

        # Markets data
        markets_db = self.data_dir / "markets.db"
        if markets_db.exists():
            try:
                with sqlite3.connect(str(markets_db)) as conn:
                    breakdown['markets_data']['count'] = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            except:
                pass

        # ML data
        ml_db = self.data_dir / "standalone_ml.db"
        if ml_db.exists():
            try:
                with sqlite3.connect(str(ml_db)) as conn:
                    breakdown['ml_experiments']['count'] = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
                    breakdown['ml_models']['count'] = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
                    breakdown['workflows']['count'] = conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0]
            except:
                pass

        # Reports
        reports = list(Path(".").glob("workflow_report_*.json"))
        breakdown['reports']['count'] = len(reports)

        # Memory
        memory_db = self.data_dir / "memory.db"
        if memory_db.exists():
            breakdown['memory']['count'] = 1  # Exists indicator

        return breakdown

    def create_vector_stores(self) -> Dict[str, Any]:
        """
        Create vector stores for LangChain semantic search.

        Creates:
        - Markets vector store: Search over market questions and descriptions
        - ML Results vector store: Search over experiments and model results
        - Combined vector store: Unified search across all data
        """
        print("🔍 Creating vector stores for LangChain integration...")

        stores_created = {}

        # 1. Markets vector store
        print("   📊 Creating markets vector store...")
        markets_store = self._create_markets_vector_store()
        if markets_store:
            stores_created['markets'] = {
                'documents': markets_store['doc_count'],
                'path': str(self.vector_store_dir / 'markets_vectorstore')
            }
            markets_store['store'].save_local(str(self.vector_store_dir / 'markets_vectorstore'))

        # 2. ML results vector store
        print("   🤖 Creating ML results vector store...")
        ml_store = self._create_ml_vector_store()
        if ml_store:
            stores_created['ml_results'] = {
                'documents': ml_store['doc_count'],
                'path': str(self.vector_store_dir / 'ml_results_vectorstore')
            }
            ml_store['store'].save_local(str(self.vector_store_dir / 'ml_results_vectorstore'))

        # 3. Combined vector store
        print("   🔗 Creating combined vector store...")
        combined_store = self._create_combined_vector_store()
        if combined_store:
            stores_created['combined'] = {
                'documents': combined_store['doc_count'],
                'path': str(self.vector_store_dir / 'combined_vectorstore')
            }
            combined_store['store'].save_local(str(self.vector_store_dir / 'combined_vectorstore'))

        print(f"✅ Created {len(stores_created)} vector stores")
        return stores_created

    def _create_markets_vector_store(self) -> Optional[Dict[str, Any]]:
        """Create vector store for market data."""
        markets_db = self.data_dir / "markets.db"
        if not markets_db.exists():
            return None

        try:
            # Load market data
            with sqlite3.connect(str(markets_db)) as conn:
                df = pd.read_sql_query("""
                    SELECT id, question, description, category, volume, active
                    FROM markets
                    WHERE active = 1 AND volume > 1000
                    LIMIT 5000  -- Limit for demo, can increase for production
                """, conn)

            if df.empty:
                return None

            # Create documents for vector store
            documents = []
            for _, row in df.iterrows():
                content = f"""
                Market Question: {row['question']}
                Description: {row['description'] or 'No description'}
                Category: {row['category']}
                Volume: ${row['volume']:,.0f}
                """.strip()

                metadata = {
                    'market_id': row['id'],
                    'category': row['category'],
                    'volume': float(row['volume']),
                    'active': bool(row['active']),
                    'data_type': 'market'
                }

                documents.append(Document(page_content=content, metadata=metadata))

            # Create vector store
            vectorstore = FAISS.from_documents(documents, self.embeddings)

            return {
                'store': vectorstore,
                'doc_count': len(documents),
                'sample_doc': documents[0] if documents else None
            }

        except Exception as e:
            print(f"❌ Error creating markets vector store: {e}")
            return None

    def _create_ml_vector_store(self) -> Optional[Dict[str, Any]]:
        """Create vector store for ML experiments and models."""
        ml_db = self.data_dir / "standalone_ml.db"
        if not ml_db.exists():
            return None

        try:
            documents = []

            with sqlite3.connect(str(ml_db)) as conn:
                # Experiments
                experiments = conn.execute("SELECT * FROM experiments").fetchall()
                for exp in experiments:
                    exp_dict = dict(zip(['experiment_id', 'workflow_id', 'name', 'phase', 'status', 'results', 'created_at'], exp))

                    content = f"""
                    Experiment: {exp_dict['name']}
                    Phase: {exp_dict['phase']}
                    Status: {exp_dict['status']}
                    Results: {exp_dict['results'] or 'No results'}
                    """.strip()

                    metadata = {
                        'experiment_id': exp_dict['experiment_id'],
                        'workflow_id': exp_dict['workflow_id'],
                        'phase': exp_dict['phase'],
                        'status': exp_dict['status'],
                        'data_type': 'experiment',
                        'created_at': exp_dict['created_at']
                    }

                    documents.append(Document(page_content=content, metadata=metadata))

                # Models
                models = conn.execute("SELECT * FROM models").fetchall()
                for model in models:
                    model_dict = dict(zip(['model_id', 'experiment_id', 'name', 'type', 'metrics', 'created_at'], model))

                    content = f"""
                    Model: {model_dict['name']}
                    Type: {model_dict['type']}
                    Metrics: {model_dict['metrics'] or 'No metrics'}
                    """.strip()

                    metadata = {
                        'model_id': model_dict['model_id'],
                        'experiment_id': model_dict['experiment_id'],
                        'model_type': model_dict['type'],
                        'data_type': 'model',
                        'created_at': model_dict['created_at']
                    }

                    documents.append(Document(page_content=content, metadata=metadata))

            if not documents:
                return None

            # Create vector store
            vectorstore = FAISS.from_documents(documents, self.embeddings)

            return {
                'store': vectorstore,
                'doc_count': len(documents),
                'sample_doc': documents[0] if documents else None
            }

        except Exception as e:
            print(f"❌ Error creating ML vector store: {e}")
            return None

    def _create_combined_vector_store(self) -> Optional[Dict[str, Any]]:
        """Create combined vector store with all data types."""
        # Get individual stores
        markets_store = self._create_markets_vector_store()
        ml_store = self._create_ml_vector_store()

        if not markets_store and not ml_store:
            return None

        all_documents = []

        if markets_store:
            # We need to recreate documents since we don't have direct access
            # In production, you'd cache these
            markets_docs = self._get_markets_documents()
            all_documents.extend(markets_docs)

        if ml_store:
            ml_docs = self._get_ml_documents()
            all_documents.extend(ml_docs)

        if not all_documents:
            return None

        try:
            vectorstore = FAISS.from_documents(all_documents, self.embeddings)
            return {
                'store': vectorstore,
                'doc_count': len(all_documents),
                'sample_doc': all_documents[0] if all_documents else None
            }
        except Exception as e:
            print(f"❌ Error creating combined vector store: {e}")
            return None

    def _get_markets_documents(self) -> List[Document]:
        """Get market documents for combined store."""
        markets_db = self.data_dir / "markets.db"
        if not markets_db.exists():
            return []

        try:
            with sqlite3.connect(str(markets_db)) as conn:
                df = pd.read_sql_query("""
                    SELECT id, question, description, category, volume, active
                    FROM markets
                    WHERE active = 1 AND volume > 1000
                    LIMIT 1000  -- Smaller limit for combined store
                """, conn)

            documents = []
            for _, row in df.iterrows():
                content = f"Market: {row['question']} | Category: {row['category']} | Volume: ${row['volume']:,.0f}"
                metadata = {
                    'market_id': row['id'],
                    'category': row['category'],
                    'volume': float(row['volume']),
                    'data_type': 'market'
                }
                documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception as e:
            return []

    def _get_ml_documents(self) -> List[Document]:
        """Get ML documents for combined store."""
        ml_db = self.data_dir / "standalone_ml.db"
        if not ml_db.exists():
            return []

        try:
            documents = []
            with sqlite3.connect(str(ml_db)) as conn:
                # Experiments (summarized)
                experiments = conn.execute("SELECT experiment_id, name, phase, status FROM experiments").fetchall()
                for exp in experiments:
                    content = f"ML Experiment: {exp[1]} | Phase: {exp[2]} | Status: {exp[3]}"
                    metadata = {
                        'experiment_id': exp[0],
                        'phase': exp[2],
                        'status': exp[3],
                        'data_type': 'experiment'
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

                # Models (summarized)
                models = conn.execute("SELECT model_id, name, type FROM models").fetchall()
                for model in models:
                    content = f"ML Model: {model[1]} | Type: {model[2]}"
                    metadata = {
                        'model_id': model[0],
                        'model_type': model[2],
                        'data_type': 'model'
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception as e:
            return []

    def create_langchain_tools(self) -> Dict[str, Any]:
        """
        Create LangChain-compatible tools for data access.

        Returns tools that agents can use to query the organized data.
        """
        from langchain.tools import tool
        from langchain_community.vectorstores import FAISS

        tools = {}

        # Vector search tools
        vector_store_paths = {
            'markets': self.vector_store_dir / 'markets_vectorstore',
            'ml_results': self.vector_store_dir / 'ml_results_vectorstore',
            'combined': self.vector_store_dir / 'combined_vectorstore'
        }

        for store_name, store_path in vector_store_paths.items():
            if store_path.exists():
                try:
                    vectorstore = FAISS.load_local(str(store_path), self.embeddings, allow_dangerous_deserialization=True)

                    # Create search tool
                    @tool
                    def search_vector_store(query: str, k: int = 5) -> str:
                        """Search the vector store for relevant information."""
                        docs = vectorstore.similarity_search(query, k=k)
                        results = []
                        for doc in docs:
                            results.append({
                                'content': doc.page_content,
                                'metadata': doc.metadata,
                                'score': getattr(doc, 'score', None)
                            })
                        return json.dumps(results, indent=2)

                    # Create filtered search tool
                    @tool
                    def search_with_filter(query: str, metadata_filter: Dict[str, Any], k: int = 5) -> str:
                        """Search with metadata filtering."""
                        docs = vectorstore.similarity_search(query, filter=metadata_filter, k=k)
                        results = []
                        for doc in docs:
                            results.append({
                                'content': doc.page_content,
                                'metadata': doc.metadata
                            })
                        return json.dumps(results, indent=2)

                    tools[f'search_{store_name}'] = search_vector_store
                    tools[f'search_{store_name}_filtered'] = search_with_filter

                except Exception as e:
                    print(f"❌ Error loading {store_name} vector store: {e}")

        # Database query tools
        @tool
        def query_markets_db(sql_query: str) -> str:
            """Execute SQL query on markets database."""
            markets_db = self.data_dir / "markets.db"
            if not markets_db.exists():
                return "Markets database not found"

            try:
                with sqlite3.connect(str(markets_db)) as conn:
                    result = pd.read_sql_query(sql_query, conn)
                    return result.to_json(orient='records', indent=2)
            except Exception as e:
                return f"Query error: {e}"

        @tool
        def query_ml_db(sql_query: str) -> str:
            """Execute SQL query on ML database."""
            ml_db = self.data_dir / "standalone_ml.db"
            if not ml_db.exists():
                return "ML database not found"

            try:
                with sqlite3.connect(str(ml_db)) as conn:
                    result = pd.read_sql_query(sql_query, conn)
                    return result.to_json(orient='records', indent=2)
            except Exception as e:
                return f"Query error: {e}"

        @tool
        def get_data_inventory() -> str:
            """Get comprehensive data inventory."""
            inventory = self.get_data_inventory()
            return json.dumps(inventory, indent=2)

        tools.update({
            'query_markets_db': query_markets_db,
            'query_ml_db': query_ml_db,
            'get_data_inventory': get_data_inventory
        })

        return tools

    def get_langchain_integration_guide(self) -> str:
        """Get comprehensive guide for LangChain integration."""
        inventory = self.get_data_inventory()

        guide = f"""
# Polymarket ML Data - LangChain Integration Guide

## 📊 Current Data Inventory

**Total Data Size**: {inventory['total_size_mb']:.1f} MB

### Databases
"""

        for db_name, db_info in inventory['databases'].items():
            guide += f"""
**{db_name}** ({db_info['size_mb']:.1f} MB)
- Tables: {', '.join(db_info['tables'])}
- Records: {sum(db_info['record_counts'].values())}
"""

        guide += """
### Data Breakdown
"""

        for data_type, info in inventory['data_breakdown'].items():
            guide += f"- **{data_type.replace('_', ' ').title()}**: {info['count']} {info['description']}\n"

        guide += """

## 🔍 Vector Stores Created

The following vector stores are available for semantic search:

1. **markets_vectorstore**: Search over 5,000+ market questions and descriptions
2. **ml_results_vectorstore**: Search over ML experiments, models, and results
3. **combined_vectorstore**: Unified search across all data types

## 🛠️ LangChain Tools Available

### Vector Search Tools
- `search_markets(query, k=5)`: Semantic search over market data
- `search_ml_results(query, k=5)`: Semantic search over ML results
- `search_combined(query, k=5)`: Unified semantic search

### Database Query Tools
- `query_markets_db(sql)`: Execute SQL on markets database
- `query_ml_db(sql)`: Execute SQL on ML database
- `get_data_inventory()`: Get current data statistics

## 💡 Usage Examples

### Basic Semantic Search
```python
from langchain.agents import create_react_agent
from your_tools import search_markets, query_markets_db

agent = create_react_agent(
    llm=your_llm,
    tools=[search_markets, query_markets_db],
    prompt=your_prompt
)

# Agent can now search markets semantically and query structured data
response = agent.run("Find high-volume crypto markets about Bitcoin")
```

### ML Experiment Analysis
```python
# Agent can search for experiments and analyze results
response = agent.run("What were the best performing ML models and why?")
```

### Combined Analysis
```python
# Cross-reference market data with ML results
response = agent.run("Find markets similar to ones where our models performed well")
```

## 🚀 Scaling Considerations

**Current Limits**:
- Markets vector store: 5,000 documents (configurable)
- ML vector store: All experiments and models
- Combined store: 6,000+ documents

**Production Scaling**:
1. Increase vector store limits as data grows
2. Implement hierarchical indexing for faster retrieval
3. Add metadata filters for more precise queries
4. Consider distributed vector stores (Pinecone, Weaviate) for larger datasets

**Performance Optimization**:
1. Pre-compute common queries
2. Use metadata filtering to reduce search space
3. Implement caching for frequent lookups
4. Batch process new data additions

## 🔄 Data Update Strategy

**Automated Updates**:
1. Daily market data ingestion → Vector store updates
2. ML workflow completion → ML vector store updates
3. Weekly re-indexing for optimal performance

**Manual Updates**:
```python
organizer = PolymarketDataOrganizer()
organizer.create_vector_stores()  # Rebuild all vector stores
```

This organization provides your LangChain agents with powerful semantic search capabilities across all your Polymarket ML data while maintaining structured access to detailed information.
"""

        return guide


def main():
    """Demo the data organization system."""
    print("🔍 Polymarket Data Organization for LangChain")
    print("=" * 50)

    organizer = PolymarketDataOrganizer()

    # Get data inventory
    print("\\n📊 Analyzing current data...")
    inventory = organizer.get_data_inventory()

    print(f"Total data size: {inventory['total_size_mb']:.1f} MB")
    print(f"Databases: {len(inventory['databases'])}")
    print(f"JSON files: {len(inventory['json_files'])}")

    print("\\n📋 Data Breakdown:")
    for data_type, info in inventory['data_breakdown'].items():
        if info['count'] > 0:
            print(f"   • {data_type.replace('_', ' ').title()}: {info['count']}")

    # Note: Vector store creation requires OpenAI API key
    print("\\n💡 To create vector stores, ensure OPENAI_API_KEY is set and run:")
    print("   organizer.create_vector_stores()")

    # Show integration guide
    print("\\n📖 LangChain Integration Guide:")
    guide = organizer.get_langchain_integration_guide()
    print(guide[:1000] + "...\\n\\n[Guide truncated - full guide available via get_langchain_integration_guide()]")

    print("\\n✅ Data organization analysis complete!")
    print(f"Run with OPENAI_API_KEY to create vector stores for semantic search.")


if __name__ == "__main__":
    main()
