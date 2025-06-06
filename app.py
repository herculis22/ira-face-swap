import gradio as gr
from PIL import Image

def generate_output(face_image, prompt):
    # dummy output - replace with your model inference code
    return face_image

demo = gr.Interface(
    fn=generate_output,
    inputs=[
        gr.Image(type="pil", label="Upload Face Image"),
        gr.Textbox(label="Describe Outfit (e.g. 'red satin lingerie')")
    ],
    outputs=gr.Image(label="Output Image"),
    title="IRA Face + Lingerie Swap",
    description="Upload a face image and describe the sexy lingerie outfit. Get a realistic result!"
)

demo.launch()
