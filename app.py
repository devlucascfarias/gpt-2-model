from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = Flask(__name__)

name_model = model.name_or_path

@app.route('/', methods=['GET', 'POST'])
def index():
    output_text = ""
    input_text = "" 
    logs = []
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        temperature = float(request.form.get('temperature', 0.7))  
        max_length = int(request.form.get('max_length', 200))  

        logs.append(f'Model Name: {name_model}')
        logs.append(f'Input Text: {input_text}')
        logs.append(f'Input IDs: {input_ids.tolist()}')
        logs.append(f'Temperature: {temperature}')
        logs.append(f'Max Length: {max_length}')

        start_time = time.time()
        output = model.generate(input_ids, max_length=max_length, temperature=temperature, do_sample=True)
        end_time = time.time()

        logs.append(f'Execution Time: {end_time - start_time} seconds')
        logs.append(f'Raw Output: {output.tolist()}')

        if output.shape[0] > 0:
            output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        else:
            output_text = ""

        logs.append(f'Output Text: {output_text}')

    return render_template('index.html', input_text=input_text, output_text=output_text, name_model=name_model, logs=logs)

if __name__ == '__main__':
    app.run(debug=True)