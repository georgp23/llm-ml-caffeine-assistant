import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def explain_choices(recommendations, prompt):
    try:
        prompt = prompt
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a cafeine recommendation assistant for a caffeine recommendation system. Your sole purpose is to explain why these drinks {recommendations} are best for the users scenario. The given drinks have been statistically proven to be the top 3 caffinated beverage for the user's scenario so make sure you do NOT change the recommendation"
                },
                {
                    "role": "user",
                    "content": f"User's scenario: {prompt}"
                }
            ],
            temperature=1.0
        )
        return response.choices[0].message.content

    except Exception as e:
            return {"error": str(e)}
