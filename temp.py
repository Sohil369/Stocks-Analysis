import google.generativeai as genai

genai.configure(api_key="AIzaSyATT0mTSy9xHWOSGzT8IMeL-82xyOm4MSo")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Summarize today’s stock market trends in one sentence.")

print(response.text)