# LangChain Retrievers: VectorStore vs. Other Approaches

This README explains different ways of creating retrievers using VectorStores in LangChain. It covers concepts like `as_retriever`, `RunnableLambda`, batch operations, and more. Use this as a reference for practical implementations.

---

## 1. **Basic Setup**
Before diving into the implementations, ensure you have the following:

### Prerequisites
- **Python version**: >= 3.8
- Install LangChain:
  ```bash
  pip install langchain
  ```
- Install a vector database backend (e.g., FAISS):
  ```bash
  pip install faiss-cpu
  ```

### Sample Documents
Create a set of sample documents for demonstration purposes:
```python
from langchain.docstore.document import Document

documents = [
    Document(page_content="The Eiffel Tower is located in Paris, France."),
    Document(page_content="The Great Wall of China is a historic landmark."),
    Document(page_content="Mount Everest is the highest mountain in the world."),
]
```

### Embedding Model
Use an embedding model for vectorization:
```python
from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()
```

### Initialize a VectorStore (e.g., FAISS)
```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embedding_model)
```

---

## 2. **Using VectorStore as a Retriever**
LangChain allows you to use a `VectorStore` directly as a retriever. This is the simplest approach to retrieve documents based on similarity.

### Example: Using `as_retriever`
```python
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

query = "Where is the Eiffel Tower located?"
results = retriever.get_relevant_documents(query)
for doc in results:
    print(doc.page_content)
```
**Key Points:**
- `as_retriever` wraps the `VectorStore` into a retriever interface.
- `search_type`: Choose between `similarity`, `mmr` (Maximal Marginal Relevance), etc.
- `search_kwargs`: Customize parameters like `k` (number of documents to return).

---

## 3. **Using `RunnableLambda` for Custom Logic**
`RunnableLambda` enables you to implement custom retrieval logic. You can modify or preprocess inputs/outputs.

### Example: Custom Retriever Logic
```python
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda

def custom_retrieval_logic(query):
    # Example: Retrieve top 3 documents and filter specific content
    results = vectorstore.similarity_search(query, k=3)
    return [doc for doc in results if "landmark" in doc.page_content.lower()]

custom_retriever = RunnableLambda(custom_retrieval_logic)

query = "Tell me about landmarks."
results = custom_retriever.invoke(query)
for doc in results:
    print(doc.page_content)
```
**Key Points:**
- Use `RunnableLambda` for flexible logic beyond standard retrievers.
- Allows chaining with other `Runnable` objects in LangChain.

---

## 4. **Batch Operations for Efficiency**
For scenarios where you need to process multiple queries at once, LangChain provides batching capabilities.

### Example: Batch Retrieval
```python
queries = [
    "Where is Mount Everest?",
    "Tell me about the Eiffel Tower.",
    "What is the Great Wall of China?"
]

batch_results = retriever.get_relevant_documents_batch(queries)
for i, results in enumerate(batch_results):
    print(f"Results for query: {queries[i]}")
    for doc in results:
        print(doc.page_content)
```
**Key Points:**
- Batch operations improve performance by reducing latency in multi-query scenarios.
- Use `get_relevant_documents_batch` with retrievers for seamless batch processing.

---

## 5. **Using a Retriever in a Chain**
You can integrate retrievers into a LangChain pipeline for complex workflows. Below is an example of chaining with a retriever, a `RunnablePassthrough`, and a chat prompt template.

### Example: Retriever with Runnable Chain
```python
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI

# Define the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Create a chat prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Based on the following documents:
    {context}

    Answer the question:
    {question}
    """
)

# Define an LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create a chain combining retriever and LLM
chain = (
    retriever
    | RunnablePassthrough()  # Pass retrieved documents to the next step
    | prompt  # Use the chat prompt template
    | llm  # Generate a response using the LLM
)

# Example query
query = "Where is the Eiffel Tower located?"
response = chain.invoke({"question": query})
print(response)
```
**Key Points:**
- `RunnablePassthrough` passes data without modifications, ensuring smooth integration.
- The chain combines retrieval (`retriever`), contextualization (`prompt`), and response generation (`llm`).

---

## 6. **Comparison: Standard Retriever vs. Custom Implementations**

| **Feature**              | **`as_retriever`**                | **`RunnableLambda`**                 | **Batch Operations**               |
|--------------------------|-----------------------------------|--------------------------------------|------------------------------------|
| **Ease of Use**          | Simple setup with minimal code    | Requires custom logic implementation | Simple for multi-query scenarios   |
| **Flexibility**          | Limited to `VectorStore` methods | Fully customizable                   | Flexible for multiple queries      |
| **Performance**          | Optimized for single queries      | Depends on logic                     | Optimized for batch processing     |
| **Use Case**             | Standard retrieval workflows      | Custom filtering or transformations  | Efficient multi-query retrieval    |

---

## 7. **Additional Notes**
- **VectorStore** backends: LangChain supports FAISS, Pinecone, Weaviate, and more.
- Use **PEFT** techniques or quantization for embedding optimization if dealing with large-scale data.
- Combine retrievers with tools like LangChain pipelines for end-to-end applications.

---

Feel free to experiment with these approaches to determine what works best for your use case!
