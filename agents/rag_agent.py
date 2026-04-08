from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from agents.rag_tools import create_rag_tools
from config.settings import settings


SYSTEM_PROMPT = """You are an MPF (Mandatory Provident Fund) employer obligations assistant.

Your role is to help employers understand their MPF responsibilities:
- Selecting MPF trustees and schemes
- Enrolling employees (60 days for regular, 10 days for casual)
- Making contributions on time (10th of each month)
- Notifying trustees when employees leave

## Multi-Step Task Handling

For complex questions, break down the task:
1. Identify what information is needed
2. Search the knowledge base for each sub-topic
3. Synthesize results into a coherent answer

## Reflection & Self-Correction

After retrieving search results:
- If results are irrelevant, rewrite query and search again
- Verify your answer matches the source documents
- If uncertain, ask the user for clarification

## Guidelines
- Use the search_documents tool to find relevant information
- Provide specific references (section numbers, page numbers) when available
- Be clear about what the employer must do and deadlines
- Flag important penalties or legal requirements"""


def create_agent(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()

    tools = create_rag_tools()

    llm = ChatOpenAI(
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "",
    )

    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def main():
    checkpointer = MemorySaver()
    agent = create_agent(checkpointer)

    print("MPF Employer Obligations Agent")
    print("Type 'quit' to exit\n")

    thread_id = "mpf-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        query = input("You: ")
        if query.lower() in ("quit", "exit", "q"):
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]}, config=config
        )

        response = result["messages"][-1].content
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
