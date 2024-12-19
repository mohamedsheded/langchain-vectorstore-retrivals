# Using Retrieved Context in LangChain Chains

This README explains how to pass retrieved context into chains using `createStuffDocumentChain` and `createRetrievalChain`. These methods allow seamless integration of retrieved documents into your LangChain workflows. It also explains their functionality and provides practical examples.

---

## 1. **Prerequisites**
Ensure you have the necessary dependencies installed:

### Install LangChain and Required Libraries
```bash
pip install langchain openai faiss-cpu
```

### Sample Documents and VectorStore Setup

Create some sample documents and initialize a VectorStore:
```python
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Sample documents
documents = [
    Document(page_content="The Eiffel Tower is in Paris, France."),
    Document(page_content="The Great Wall of China is a historic wonder."),
    Document(page_content="Mount Everest is the tallest mountain in the world."),
]

# Embedding model and vectorstore
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)
```

---

## 2. **Method 1: `createStuffDocumentChain`**

The `createStuffDocumentChain` method allows you to create a chain that combines retrieved documents into a single context for processing.

### How It Works
- Retrieves documents using a retriever.
- Combines the documents into a single string or structured format.
- Passes the combined context to a language model (or other downstream steps).

### Example
```python
from langchain.chains.combine_documents import createStuffDocumentChain
from langchain.chat_models import ChatOpenAI

# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Define the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the chain
stuff_chain = createStuffDocumentChain(llm=llm, retriever=retriever)

# Run the chain
query = "Tell me about the Eiffel Tower."
response = stuff_chain.run(query)
print(response)
```

### Key Points
- Suitable for simple use cases where all retrieved documents can be merged directly.
- Requires minimal customization.

---

## 3. **Method 2: `createRetrievalChain`**

`createRetrievalChain` provides a more flexible framework for combining retrieval and language model operations.

### How It Works
- Wraps a retriever and LLM into a cohesive chain.
- Handles the retrieval, contextualization, and response generation process.
- Allows prompt customization for specific tasks.

### Example
```python
from langchain.chains import createRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Define a custom prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant. Based on the following documents:
    {context}

    Answer the question:
    {question}
    """
)

# Define the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the chain
retrieval_chain = createRetrievalChain(
    retriever=retriever, prompt=prompt, llm=llm
)

# Run the chain
query = "What is the Eiffel Tower?"
response = retrieval_chain.run({"question": query})
print(response)
```

### Key Points
- Ideal for scenarios requiring custom prompts.
- More adaptable to complex workflows.

---

## 4. **Context-Aware Retriever with Message Placeholder**

For more dynamic workflows, you can create a context-aware retriever that incorporates placeholders for interactive messaging.

### How It Works
- Retrieves documents based on user input.
- Dynamically fills placeholders in the message template with retrieved context and user queries.

### Example
```python
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import createRetrievalChain
from langchain.chat_models import ChatOpenAI

# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Define a dynamic prompt with placeholders
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant. Use the following retrieved context to help answer the user's query:
    {context}

    User's question:
    {user_input}

    Provide a detailed and helpful response.
    """
)

# Define the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the chain
context_aware_chain = createRetrievalChain(
    retriever=retriever, prompt=prompt, llm=llm
)

# Run the chain
query = "What is the tallest mountain in the world?"
response = context_aware_chain.run({"user_input": query})
print(response)
```

### Key Points
- Enables more interactive and user-specific query handling.
- Uses placeholders (`{context}`, `{user_input}`) for dynamic message construction.

---

## 5. **Comparison: `createStuffDocumentChain` vs. `createRetrievalChain` vs. Context-Aware Retriever**

| **Feature**              | **`createStuffDocumentChain`**     | **`createRetrievalChain`**             | **Context-Aware Retriever**         |
|--------------------------|-------------------------------------|----------------------------------------|-------------------------------------|
| **Ease of Use**          | Easy setup with minimal parameters | Requires prompt definition             | Requires placeholders in prompts    |
| **Flexibility**          | Limited customization              | Fully customizable with prompts        | Highly dynamic for interactive tasks |
| **Performance**          | Suitable for small-scale workflows | Optimized for complex use cases        | Best for user-specific interactions |
| **Use Case**             | Simple document merging            | Contextualized question-answering      | Interactive and dynamic responses   |

---

## 6. **Conclusion**
- Use `createStuffDocumentChain` for straightforward scenarios where retrieved documents are directly combined and passed to the LLM.
- Opt for `createRetrievalChain` for advanced use cases requiring detailed prompts and structured workflows.
- Choose the context-aware retriever for highly dynamic and interactive applications.

Each method provides powerful ways to integrate retrieval and context-aware processing in LangChain workflows. Select based on your specific requirements!
