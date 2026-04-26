#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
from typing import Optional
import litellm

class LLMClient:
    """LLM client class for Duke LiteLLM / OpenAI"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://litellm.oit.duke.edu",
        model: str = "gpt-5-mini",
        timeout: int = 300,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter: bool = True,
    ):
        # Use the provided key or look for the Duke-specific environment variable
        self.api_key = api_key or os.getenv("DUKE_LITELLM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Duke LiteLLM API key not found. "
                "Set DUKE_LITELLM_API_KEY or pass api_key explicitly."
            )

        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs,) -> str:
        """Generate text using Duke LiteLLM"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "messages": messages
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens:
            params["max_tokens"] = max_tokens

        return self._generate_with_retry(**params)

    def _generate_with_retry(self, **params) -> str:
        model_name = self.model 

        temperature = params.get("temperature", None)

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=model_name,
                    messages=params["messages"],
                    base_url=self.base_url, 
                    api_key=self.api_key,
                    temperature=temperature if temperature is not None else 1.0,
                    max_tokens=params.get("max_tokens"),
                    timeout=self.timeout,
                )
                
                return response.choices[0].message.content

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.retry_base_delay * (self.retry_backoff_factor ** attempt),
                        self.retry_max_delay,
                    )
                    if self.retry_jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    print(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries. Last error: {last_exception}"
        )
    

def create_client(
    api_key: Optional[str] = None, 
    base_url: str = "https://litellm.oit.duke.edu", 
    model: str = "gpt-5-mini"
) -> LLMClient:
    """Helper function to initialize the LLMClient"""
    return LLMClient(api_key=api_key, base_url=base_url, model=model)


# Example usage
if __name__ == "__main__":

    # Method 2: Custom base_url (e.g., using OpenAI)
    # client = create_client(
    #     api_key=os.getenv("DUKE_LITELLM_API_KEY"),       # Duke key
    #     base_url="https://litellm.oit.duke.edu",  # LiteLLM proxy
    #     model="gpt-5-mini"                        # LiteLLM model
    # )

#     # Method 3: Use API key from environment variable
#     # export OPENAI_API_KEY="your-api-key"
#     # client = create_client()

    # Generate content
    response = litellm.completion(
        model="gpt-5-mini",
        messages=messages,
        base_url="https://litellm.oit.duke.edu",
        api_key=os.getenv("DUKE_LITELLM_API_KEY"),
    )
    print(response.choices[0].message.content)

    # Stream generation
    # for chunk in client.generate_stream(prompt="Write a poem about spring"):
    #     print(chunk, end="", flush=True)
