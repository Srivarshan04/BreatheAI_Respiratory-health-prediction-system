import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sys 
load_dotenv()
api_key=os.environ.get("GEMINI_API_KEY")

client=genai.Client(api_key=api_key)

if len(sys.argv)<2:
    print("i need a prompt")
    sys.exit(1)
prompt=sys.argv[1]
messages=[types.Content(role="user", path=[types.Part(ext=prompt)])]

response=client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)

print(response.text)
print(f"prompt tokens:{response.usage_metadata.prompt_token_count}")
print(f"response tokens:{response.usage_metadata.candidates_token_count}")