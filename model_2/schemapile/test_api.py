import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Initialize client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Test
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tell me about yourself!"
)
print(response.text)