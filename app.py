import torch
import gradio as gr
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('clip-ViT-L-14')

def predict(im1, im2):
  embeding = model.encode([im1, im2])
  sim = cosine_similarity(embeding)
  sim = sim[0][1]
  if sim > 0.75:
    return sim, "SAME PERSON, UNLOCK PHONE"
  else:
    return sim, "DIFFERENT PEOPLE, DON'T UNLOCK"


title="Face-id Application Demo"
description = "Upload similar/different images to compare Image similarity for face-id demo"
article = """
            - Select an image from the examples provided as demo image
            - Click submit button to make Image classification
            - Click clear button to try new Image for classification
          """

img_upload = gr.Interface(
    fn=predict, 
    inputs= [gr.Image(type="pil", source="upload"), 
             gr.Image(type="pil", source="upload")], 
    outputs= [gr.Number(label="Similarity"),
              gr.Textbox(label="Message")],
    title=title,
    description=description,
    article=article,
    examples=[['examples/img1.jpg', 'examples/img2.jpg'],
              ['examples/img1.jpg', 'examples/img3.jpg']]
    )

webcam_upload = gr.Interface(
    fn=predict, 
    inputs= [gr.Image(type="pil", source="webcam"), 
            gr.Image(type="pil", source="webcam")], 
    outputs= [gr.Number(label="Similarity"),
              gr.Textbox(label="Message")],
    title=title,
    description=description,
    article=article,
    )

face_id = gr.TabbedInterface(
    [img_upload, webcam_upload], 
    ["Upload-Image", "Use Webcam"])

face_id.launch(debug=True)
