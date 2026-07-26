from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import requests

import streamlit as st

st.title("PlayList Generator")
user_prompt = st.text_input("enter the prompt or the name of the song")
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("API")


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key,
)

API_KEY = "d3f81ae1c4e2ac5200168c552ff912cc"


@tool
def search_song(query: str) -> str:
    """
    Search Last.fm for songs.
    """

    url = "https://ws.audioscrobbler.com/2.0/"

    params = {
        "method": "track.search",
        "track": query,
        "api_key": API_KEY,
        "format": "json",
        "limit": 5
    }

    response = requests.get(url, params=params)

    data = response.json()

    songs = data["results"]["trackmatches"]["track"]

    result = []

    for song in songs:
        result.append(
            f"{song['name']} by {song['artist']}"
        )

    return "\n".join(result)


agent = create_agent(
    model=llm,
    tools=[search_song]
)



if user_prompt:

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
    You are a music recommendation assistant.

    Use the search_song tool whenever required.

    User request:

    {user_prompt}

    Create the best playlist.
    """
            }
        ]
    }

)
if user_prompt:
    st.markdown(response["messages"][-1].content)

