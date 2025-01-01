import google.generativeai as genai

# Configure the Gemini API with your API key
genai.configure(api_key="AIzaSyAEOVGaFTlqhwDVhuw5XuddBOM6ZNoYIdk")

async def generate_heading(user_message: str, ai_response: str) -> str:
    try:
        # Prepare the prompt for generating the heading
        prompt = f"""
        User's message: {user_message}
        AI's response: {ai_response}

        Based on the above, generate a very very short, Title for this conversation:
        Note: Please don't use any commas and any type of brackets
        """
        
        # Use the generative model to create the content
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Extract the text from the response and truncate to 50 characters
        heading = response.text.strip()[:50]
        return heading

    except Exception as e:
        print(f"Error in generate_heading: {e}")
        return "Default Heading"
