
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import logging

load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

INFO_BURIED = [
]
# Sample knowledge base for demos
TECH_DOCS = [
]

def create_base_vectorstore():
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )

def demo_multi_query_retriever():
    print("=" * 60)
    print("MULTI-QUERY RETRIEVER")
    print("Generates multiple perspectives on your question")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), llm=llm
    )

    query = "What tools can I use to build AI applications?"

    print(f"\nOriginal Query: {query}")
    print("\nThe retriever will generate multiple query variations...")
    print("(Check INFO logs above for generated queries)\n")

    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} unique documents:")

    for i, doc in enumerate(docs):
        print(
            f"\n{i + 1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )

def demo_contextual_compression():
    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extracts only query-relevant content from documents")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exist for building LLM applications?"

    print(f"\nQuery: {query}")

    # Without compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")

    # With compression
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content}\n")


def demo_ensemble_hybrid_search():
    print("=" * 60)
    print("ENSEMBLE/HYBRID RETRIEVER")
    print("Combines keyword (BM25) + semantic search")
    print("=" * 60)

    vectorstore = create_base_vectorstore()

    bm25_retriever = BM25Retriever.from_documents(TECH_DOCS)
    bm25_retriever.k = 3

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
    )

    queries = [
        "ACID transactions",  # Keyword-heavy (BM25 helps)
        "How do I store AI model outputs for later retrieval?",  # Semantic (vectors help)
        "fast similarity lookup for embeddings",  # Mixed
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        # Compare results
        bm25_results = bm25_retriever.invoke(query)
        semantic_results = semantic_retriever.invoke(query)
        ensemble_results = ensemble_retriever.invoke(query)

        print(f"BM25 top result: {bm25_results[0].page_content[:60]}...")
        print(f"Semantic top result: {semantic_results[0].page_content[:60]}...")
        print(f"Ensemble top result: {ensemble_results[0].page_content[:60]}...")

def demo_parent_document_retriever():
    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("=" * 60)

    long_doc = Document(
        page_content="""
    # Complete Guide to Building AI Agents

    ## Chapter 1: Introduction to AI Agents

    AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

    The key components of an AI agent include:
    - A language model for reasoning
    - Tools for interacting with external systems
    - Memory for maintaining context
    - A planning mechanism for complex tasks

    ## Chapter 2: Agent Frameworks

    Several frameworks exist for building AI agents:

    LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

    LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

    CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

    ## Chapter 3: Production Considerations

    Deploying agents to production requires careful attention to:
    - Error handling and fallbacks
    - Token usage optimization
    - Observability and tracing
    - Security and access control
    - State persistence and recovery

    LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
            """,
        metadata={"source": "ai_agents_guide.md"},
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents([long_doc])

    # Search
    query = "What is LangGraph used for?"

    print(f"\nQuery: {query}")

    # Regular retrieval (would get small chunks)
    child_docs = vectorstore.similarity_search(query, k=1)
    print(f"\n--- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content[:300]}...")


def demo_advanced_rag_chain():
    print("=" * 60)
    print("COMPLETE ADVANCED RAG CHAIN")
    print("Multi-query + Compression + RAG")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), llm=llm
    )

    compressor = LLMChainExtractor.from_llm(llm)
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_retriever
    )

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the question based on the following context. Be specific and cite which technologies you're referring to.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What options do I have for building AI agents?",
        "How can I store and search embeddings?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")

if __name__ == "__main__":
    # demo_multi_query_retriever()
    # demo_contextual_compression()
    # demo_ensemble_hybrid_search()
    # demo_parent_document_retriever()
    demo_advanced_rag_chain()