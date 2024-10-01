import gradio as gr
from chatbot import Chatbot

chatbot = Chatbot("meta-llama/Llama-3.2-1B-Instruct")

def get_pdf(pdf_file):
    print("Submitted PDF", pdf_file)
    file_name = str(pdf_file)
    chatbot.load_db(file_name)

def responde(message, history):
    ret = chatbot.respond(message)
    return ret

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>RAG Chatbot</h1>")

    pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
    submit_pdf_btn = gr.Button("Submit PDF")
    submit_pdf_btn.click(fn=get_pdf, inputs=[pdf_file])

    gr.ChatInterface(responde, chatbot=gr.Chatbot(height=400))

demo.launch()

