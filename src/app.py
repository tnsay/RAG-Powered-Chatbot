import gradio as gr
from rag_pipeline import run_rag_pipeline

def query_rag(question):
    answer, sources = run_rag_pipeline(question)
    source_text = "\n\n---\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(sources)])
    return answer, source_text

with gr.Blocks() as demo:
    gr.Markdown("## RAG-Powered Complaint Assistant")

    with gr.Row():
        inp = gr.Textbox(lines=2, placeholder="Ask a question about a customer complaint...")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=6)
    with gr.Row():
        source_box = gr.Textbox(label="Source Chunks", lines=10)

    submit_btn.click(query_rag, inputs=inp, outputs=[answer_box, source_box])
    clear_btn.click(lambda: ("", ""), None, [answer_box, source_box])

demo.launch()
