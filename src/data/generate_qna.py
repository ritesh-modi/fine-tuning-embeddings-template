from openai import AzureOpenAI
import pandas as pd
import os
from config_loader import DATASET_CONFIGS
# Replace with your valid Google API key
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),  
    azure_deployment=os.getenv("AZURE_DEPLOYMENT")
)
model_name = os.getenv("MODEL_NAME")
temperature = os.getenv("TEMPERATURE")



# Prompt generator with an explicit request for structured output
def prompt(text_chunk, dataset_format):
    #return PROMPTS[dataset_format].format(text_chunk=text_chunk)
    return DATASET_CONFIGS[dataset_format]['prompt'].format(text_chunk=text_chunk)

# Function to interact with OpenAI and return a QA pair
def generate_with_openai(text_chunk:str, dataset_format:str):
    try:
        response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant and expert question and answer generator."},
                    {"role": "user", "content": prompt(text_chunk, dataset_format)}
                ], temperature=0.0
            )
        content = response.choices[0].message.content.strip()

        split_keys = DATASET_CONFIGS[dataset_format]['split_keys']
        parts = content.split('\n')
        result = {}
        for key, part in zip(split_keys, parts):
            result[key.lower().rstrip(':')] = part.lower().replace(key.lower(), '').strip()
        return result
    except Exception as e:
        return {"error": str(e)}

def generate_qna(text_chunks, dataset_format):
    results = []
    expected_columns = len(DATASET_CONFIGS[dataset_format]['columns'])
    # Iterate through each text chunk
    for chunk in text_chunks:
        result = generate_with_openai(chunk, dataset_format)
        if "error" not in result:
            results.append(result)
        else:
            print(f"Error generating data for chunk: {chunk[:50]}...")  # Print first 50 chars of problematic chunk

    #final_results = [d for d in results if all(value is not None for value in d.values())]
    
    i = 0
    while i < len(results):
        item = results[i]
        if all(value is not None for value in item.values()) and len(item) == expected_columns:
            i += 1
        else:
            print(f"Removing item due to None values or mismatched column count: {item}")
            results.pop(i)
    # Convert results into a Pandas DataFrame
    df = pd.DataFrame(results)
    df = df.rename(columns=DATASET_CONFIGS[dataset_format]['columns'])
    return df