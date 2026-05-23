from pprint import pprint

from src.agents_src.crew import qa_crew



result = qa_crew.kickoff(
    inputs = {
        "user_query": "Explain about operating systems?",
        "chat_history": "{}"
    }
)

result_dict = result.to_dict()
pprint(result_dict)