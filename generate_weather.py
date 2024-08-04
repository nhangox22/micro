from transformers import pipeline
import pandas as pd


generator = pipeline('text-generation', model='gpt2')


cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
seasons = ["Summer", "Autumn", "Winter", "Spring"]


def generate_weather(city, season):
    prompt = f"What is the weather like in {city} during {season}?"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    return result[0]['generated_text'].split('?')[1].strip()  


weather_df = pd.DataFrame(columns=cities, index=seasons)


for city in cities:
    for season in seasons:
        weather_df.at[season, city] = generate_weather(city, season)


weather_df.to_csv('australian_weather_by_season.csv')


print(weather_df)
