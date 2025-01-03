import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

async def generate_heading(user_message: str, ai_response: str) -> str:
    try:
        prompt = f"""
        User's message: {user_message}
        AI's response: {ai_response}

        Based on the above, generate a very very short, Title for this conversation:
        Note: Please don't use any commas and any type of brackets
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        heading = response.text.strip()[:50]
        return heading

    except Exception as e:
        print(f"Error in generate_heading: {e}")
        return "Default Heading"
