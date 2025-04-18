# Blog Writer
import os
os.environ["TOGETHER_API_KEY"] = "1ca4ccbd327207e207123e974f062b36ada5ac64d78fe6e27cbf41f68076b9a1"

#importing of the required libraries

import asyncio
import aiohttp
import os
import json
import re
import requests
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from functools import lru_cache
from textstat import flesch_reading_ease
from rich.console import Console
from dotenv import load_dotenv
import streamlit as st

console = Console()
load_dotenv()

# Configuration using together api
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
NEWSDATA_API = os.getenv("NEWSDATA_API")

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Analysis of the topic
def analyze_topic(topic: str, tone: str) -> List[str]:
    base = re.split(r'[:\-]', topic)[0].strip()
    return [f"Introduction to {base}", f"Applications of {base}", f"Future of {base}"]

#Gathering of the contexts by the use of asyncio for concurrent API requests 
async def fetch_json(session, url, retries=3):
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return await response.json()
                else:
                    text = await response.text()
                    raise ValueError(f"❌ Non-JSON response ({response.status}):\n{text}")
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)
            else:
                raise e
# Use of caching
@lru_cache(maxsize=64)
async def fetch_research_data(topic: str):
    async with aiohttp.ClientSession() as session:
        news_url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API}&q={topic}&language=en"
        datamuse_url = f"https://api.datamuse.com/words?ml={topic}"
        quote_url = f"https://api.api-ninjas.com/v1/quotes?query={topic}"


        results = {}
        errors = {}

        async def try_fetch(key, url):
            try:
                results[key] = await fetch_json(session, url)
            except Exception as e:
                errors[key] = str(e)
                results[key] = None

        await asyncio.gather(
            try_fetch("news", news_url),
            try_fetch("keywords", datamuse_url),
            try_fetch("quotes", quote_url)
        )

        #Log warnings in console
        for key, err in errors.items():
          console.print(f"[yellow]⚠️ Warning: Failed to fetch {key}: {err}[/yellow]")
          st.warning(f"⚠️ Warning: Failed to fetch {key} data. The blog may lack context.")

        return {
            "news": results["news"].get("results", []) if results["news"] else [],
            "keywords": [kw["word"] for kw in results["keywords"][:10]] if results["keywords"] else [],
            "quotes": [q["content"] for q in results["quotes"].get("results", [])[:3]] if results["quotes"] else []
        }


# CONTENT GENERATION 
def gpt_call(prompt, tone):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant that writes {tone} blog posts."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return f"⚠️ GPT API error ({response.status_code}): {response.text}"
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return "⚠️ Empty content returned by model. Please try again later."
        return content
    except Exception as e:
        return f"⚠️ Exception while calling GPT API: {e}"

async def generate_blog_content(topic: str, subtopics: List[str], context_data: Dict, tone: str):
    def prompt_for(subtopic):
        keywords = ", ".join(context_data["keywords"][:5])
        quote_block = "\n".join([f"> {q}" for q in context_data["quotes"]]) if context_data["quotes"] else ""
        return (
            f"Write a ~250 word blog section in a {tone} tone on the topic '{subtopic}'. "
            f"Use the following keywords for SEO: {keywords}. Include references to these quotes if useful:\n{quote_block}"
        )

    intro_prompt = f"Write an engaging introduction (~150 words) for a blog titled '{topic}' in a {tone} tone."
    conclusion_prompt = f"Write a strong conclusion (~100 words) for a blog titled '{topic}', ending with a call-to-action."

    tasks = [
        asyncio.to_thread(gpt_call, intro_prompt, tone),
        *(asyncio.to_thread(gpt_call, prompt_for(sub), tone) for sub in subtopics),
        asyncio.to_thread(gpt_call, conclusion_prompt, tone)
    ]

    results = await asyncio.gather(*tasks)
    intro, *bodies, conclusion = results

    # Fallbacks for failed sections
    intro = intro if "⚠️" not in intro else "We're excited to explore this topic with you. Let's dive in!"
    conclusion = conclusion if "⚠️" not in conclusion else "Thanks for reading! Stay tuned for more insights."

    for i in range(len(bodies)):
        if "⚠️" in bodies[i]:
            bodies[i] = f"Sorry, we couldn't generate this section due to API issues."

    body = ""
    for sub, content in zip(subtopics, bodies):
        body += f"\n\n## {sub}\n\n{content}"

    blog_md = f"## Introduction\n\n{intro}" + body + f"\n\n## Conclusion\n\n{conclusion}"
    return blog_md, len(blog_md.split())

# SEO METADATA 
def generate_seo_metadata(topic: str, content: str, word_count: int) -> Dict:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    meta = {
        "title": topic.title(),
        "description": content.split("\n\n")[1][:160],
        "keywords": list(set(topic.lower().split())),
        "slug": slug,
        "read_time": max(1, word_count // 200),
        "readability": flesch_reading_ease(content)
    }
    return meta

# Export
def export_blog(md_path: Path, json_path: Path, blog_md: str, metadata: Dict):
    with open(md_path, "w") as f:
        f.write(blog_md)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

# STREAMLIT UI 
st.title("🧠 AI Blog Writer Agent")
topic = st.text_input("Enter your blog topic:")
tone = st.selectbox("Select tone (or we’ll choose one for you):", ["", "Educational", "Conversational", "Professional", "Inspirational", "Neutral"])

if st.button("Generate Blog") and topic:
    if not tone:
        tone = random.choice(["Educational", "Conversational", "Professional", "Inspirational", "Neutral"])
        st.info(f"No tone selected — defaulting to: {tone}")

    subtopics = analyze_topic(topic, tone)
    try:
        context_data = asyncio.run(fetch_research_data(topic))
    except Exception as e:
        st.error(f"🚨 Failed to gather context data: {e}")
        st.stop()

    blog_md, word_count = asyncio.run(generate_blog_content(topic, subtopics, context_data, tone))
    seo_metadata = generate_seo_metadata(topic, blog_md, word_count)

    slug = seo_metadata["slug"]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    md_path = output_dir / f"{slug}.md"
    json_path = output_dir / f"{slug}.json"
    export_blog(md_path, json_path, blog_md, seo_metadata)

    st.success("✅ Blog generated!")
    st.markdown("### 📄 Blog Content")
    st.markdown(blog_md)
    st.markdown("### 📑 SEO Metadata")
    st.json(seo_metadata)
