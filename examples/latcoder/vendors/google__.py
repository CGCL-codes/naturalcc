from google import genai

def gemini(prompt, texts_imgs=[], temperature=0, seed=0, n=1):
    generation_config = {"temperature":temperature}
    client = genai.Client(api_key='xx-xx')
    try:
        response=client.models.generate_content(model='gemini-1.5-pro-latest', contents=[prompt]+texts_imgs, config=generation_config)
        return response.text
    except Exception as e:
        print("Error during content generation:", e)
