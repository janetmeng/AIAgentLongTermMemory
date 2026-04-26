import litellm
import os

messages = [
    {
        "role": "user",
        "content": "what llm are you"
    }
]                          
response = litellm.completion(
        model="gpt-5-mini",
        messages=messages,
        base_url="https://litellm.oit.duke.edu",
        api_key=os.getenv("DUKE_LITELLM_API_KEY"),
    )

print(response.choices[0].message.content)