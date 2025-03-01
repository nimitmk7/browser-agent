from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller
import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key = os.getenv("GEMINI_API_KEY"))

def initialize_agent(query):
    llm = get_llm()
    controller = Controller()
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

    async def run_agent():
        with st.spinner("Running Agent..."):
            await agent.run(max_steps=25)
        st.success("Agent completed Task successfully!")
    
    asyncio.run(run_agent())

    st.button("Close Browser", on_click=lambda: asyncio.run(browser.close()))

