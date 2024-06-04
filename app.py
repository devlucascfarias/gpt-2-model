from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output_text = ""
    logs = []
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Add input_text and input_ids to logs
        logs.append(f'Input Text: {input_text}')
        logs.append(f'Input IDs: {input_ids.tolist()}')

        # Generate text
        output = model.generate(input_ids, max_length=200, temperature=0.7, do_sample=True)

        # Add raw output to logs
        logs.append(f'Raw Output: {output.tolist()}')

        # Decode the output
        output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Add output_text to logs
        logs.append(f'Output Text: {output_text}')

    return render_template('index.html', output_text=output_text, logs=logs)

if __name__ == '__main__':
    app.run(debug=True)