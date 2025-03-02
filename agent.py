from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from typing import List

from pydantic import BaseModel

class News(BaseModel):
	headline: str
	url: str


class NewsList(BaseModel):
	news: List[News]

def get_llm():
    """
    Returns an instance of the Google Generative AI LLM.

    The LLM is initialized with the "gemini-1.5-flash" model and the API key
    set in the GEMINI_API_KEY environment variable.

    Returns:
        llm (ChatGoogleGenerativeAI): An instance of the Google Generative AI LLM.
    """
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key = os.getenv("GEMINI_API_KEY"))

def initialize_agent(query):
    """
    Initializes a new instance of the Agent class with the given query.

    This function also initializes a new instance of the Controller and Browser classes
    and returns them as a tuple.

    Args:
        query (str): The query to use for the Agent.

    Returns:
        tuple: A tuple containing the initialized Agent and Browser.
    """
    llm = get_llm()
    controller = Controller(output_model=News)
    browser = Browser(config=BrowserConfig())

    return Agent(
        task = query,
        llm = llm, 
        controller = controller,
        browser = browser, 
        use_vision=True, 
        max_actions_per_step=1
    ), browser

st.title("Browser AI Agent 🤖 powered by Google Gemini")

query = st.text_input("Enter your query here:")

if st.button("Run Agent"):
    st.write("Initlializing Agent...")
    agent, browser = initialize_agent(query)

    # Create a placeholder to update results dynamically
    output_area = st.empty()

    async def run_agent():
        with st.spinner("Running Agent..."):
            history = await agent.run(max_steps=25)
            
            result = history.final_result()
            
            output_area.write("### Results from Agent:")
            print(result)
            if result:
                parsed: NewsList = NewsList.model_validate_json(result)
                for news in parsed.news:
                    output_area.write(f"Title: {news.headline}")
                    output_area.write(f"URL: {news.url}")
            else:
                output_area.write("No results found")

        st.success("Agent completed Task successfully!")
    
    asyncio.run(run_agent())



