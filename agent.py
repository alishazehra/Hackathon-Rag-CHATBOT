# from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
# from agents import set_tracing_disabled, function_tool
# import os
# from dotenv import load_dotenv
# from agents import enable_verbose_stdout_logging

# enable_verbose_stdout_logging()

# load_dotenv()
# set_tracing_disabled(disabled=True)

# gemini_api_key = os.getenv("OPENAI_API_KEY")
# provider = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# model = OpenAIChatCompletionsModel(
#     model="gpt-4o-mini",
#     openai_client=provider
# )

# import cohere
# from qdrant_client import QdrantClient

# # Initialize Cohere client
# cohere_client = cohere.Client("key-here")
# # Connect to Qdrant
# qdrant = QdrantClient(
#     url="url-here",
#     api_key="here-key" 
# )



# def get_embedding(text):
#     """Get embedding vector from Cohere Embed v3"""
#     response = cohere_client.embed(
#         model="embed-english-v3.0",
#         input_type="search_query",  # Use search_query for queries
#         texts=[text],
#     )
#     return response.embeddings[0]  # Return the first embedding


# @function_tool
# def retrieve(query):
#     embedding = get_embedding(query)
#     result = qdrant.query_points(
#         collection_name="humanoid_ai_book",
#         query=embedding,
#         limit=5
#     )
#     return [point.payload["text"] for point in result.points]



# agent = Agent(
#     name="Assistant",
#     instructions="""
# You are an AI tutor for the Physical AI & Humanoid Robotics textbook.
# To answer the user question, first call the tool `retrieve` with the user query.
# Use ONLY the returned content from `retrieve` to answer.
# If the answer is not in the retrieved content, say "I don't know".
# """,
#     model=model,
#     tools=[retrieve]
# )


# result = Runner.run_sync(
#     agent,
#     input="what is physical ai?",
# )

# print(result.final_output)


from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool, enable_verbose_stdout_logging
import os
from dotenv import load_dotenv

# ------------------------------------
# Setup
# ------------------------------------
# enable_verbose_stdout_logging()
load_dotenv()
set_tracing_disabled(disabled=True)

# ------------------------------------
# OpenAI provider (NO base_url needed)
# ------------------------------------
provider = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------------------------
# Model: GPT-4o Mini
# ------------------------------------
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=provider
)

# ------------------------------------
# Cohere + Qdrant
# ------------------------------------
import cohere
from qdrant_client import QdrantClient

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# ------------------------------------
# Embedding function
# ------------------------------------
def get_embedding(text: str):
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[text],
    )
    return response.embeddings[0]

# ------------------------------------
# RAG retrieve tool
# ------------------------------------
@function_tool
def retrieve(query: str):
    embedding = get_embedding(query)

    result = qdrant.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5
    )

    return [point.payload["text"] for point in result.points]

# ------------------------------------
# Agent
# ------------------------------------
agent = Agent(
    name="Assistant",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook. 
If the user greets you like (hi, hello) respond politely and ask them to talk about the book also handle their queries 

To answer the user question:
1. FIRST call the tool `retrieve` with the user query.
2. Use ONLY the returned content from `retrieve` to answer.
3. If the answer is not in the retrieved content, List exactly:
   " The book of Physical AI and Robotics are vast technologies to begin your journey"
""",
    model=model,
    tools=[retrieve]
)

# ------------------------------------
# Run
# ------------------------------------
def ask_agent(user_question: str):
    result = Runner.run_sync(
        agent,
        input=user_question,
    )
    return result.final_output