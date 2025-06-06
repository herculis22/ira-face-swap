import gradio as gr
from predict import Predictor

predictor = Predictor()
predictor.setup()

def swap_face(prompt, face_image):
    output_path = predictor.predict(prompt=prompt, face_image=face_image)
    return str(output_path)

iface = gr.Interface(
    fn=swap_face,
    inputs=[
        gr.Textbox(label="Prompt (e.g., Woman in red lingerie)"),
        gr.Image(type="filepath", label="Upload Face Image")
    ],
    outputs=gr.Image(label="Result"),
    title="IRA Face Swap AI",
    description="Upload a face image and type a prompt. The AI will generate a dressed version."
)

if __name__ == "__main__":
    iface.launch()
