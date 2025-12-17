# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.agents import initialize_agent, AgentType
# from langchain.llms import Ollama
# from langchain.tools import Tool

# # Initialize Ollama model
# llm = Ollama(model="llama2:7b")

# # Setup Wikipedia search
# wiki_wrapper = WikipediaAPIWrapper()
# wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# # Create tools list
# tools = [
#     Tool.from_function(
#         func=wiki_tool.run,
#         name="Wikipedia Search",
#         description="Useful for searching Wikipedia for factual information"
#     )
# ]

# # Create agent
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# # Run agent
# response = agent.run("Tell me about recent AI developments")
# print(response)



from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from src.llm_factory import LLMFactory, LLMProvider
from src.llm_langchain_wrapper import LangChainLLMWrapper

# Import all callers to register them with the factory
import gemini_caller
import qwen_caller

# Initialize LLM using factory (you can choose: gemini, qwen, or glm)
# Example with Qwen:
llm_caller = LLMFactory.create_caller(
    provider=LLMProvider.QWEN,
    api_key="sk-fc88e8c463e94a43bc41f1094a28fa1f",
    model="qwen3-max"
)
llm = LangChainLLMWrapper(llm_caller=llm_caller)

# Or use Gemini:
# llm_caller = LLMFactory.create_caller(
#     provider=LLMProvider.GEMINI,
#     api_key="AIzaSyAt19tBj232GyyUbM95MlZzZarqZcTKmsc",
#     model="gemini-2.5-flash"
# )
# llm = LangChainLLMWrapper(llm_caller=llm_caller)

# Or use GLM:
# llm_caller = LLMFactory.create_caller(
#     provider=LLMProvider.GLM,
#     api_key="0aa9ad7dd2114ef3bfb16bea056a977a.9nVmCTUa8l6SBCK6",
#     model="glm-4.6"
# )
# llm = LangChainLLMWrapper(llm_caller=llm_caller)

# Create multiple tools
search_tool = DuckDuckGoSearchTool()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Custom calculator tool
def calculator(query):
    try:
        return str(eval(query))
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# Combine tools
tools = [
    # Tool.from_function(
    #     func=search_tool.run,
    #     name="Web Search",
    #     description="Useful for searching current information on the web"
    # ),
    Tool.from_function(
        func=wiki_tool.run,
        name="Wikipedia Search",
        description="Useful for searching factual information on Wikipedia"
    ),
    Tool.from_function(
        func=calculator,
        name="Calculator",
        description="Useful for performing mathematical calculations"
    )
]

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example usage
response = agent.run("What is the population of USA, what is the unemployed rate in 2025? calculate how many people possibly lost their jobs in 2025 in USA.")
print(response)
