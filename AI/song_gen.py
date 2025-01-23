from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route('/generate', methods=['POST'])
def generate_song():
    data = request.json
    prompt = data.get('prompt', '')
    # Generate song lyrics using the fine-tuned model
    result = generator(prompt, max_length=500, num_return_sequences=1, truncation=True)
    return jsonify({"song": result[0]['generated_text']})

if __name__ == '__main__':
    app.run(port=5000)
