import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# load model
model = tf.keras.models.load_model("pneumonia_model.keras")

def predict(image):
    # resize to 224x224
    image = image.resize((224, 224))
    
    # convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # handle grayscale X-rays (convert to RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:,:,:3]
    
    # add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # predict
    prediction = model.predict(img_array)[0][0]
    
    # return probabilities
    return {
        "PNEUMONIA": float(prediction),
        "NORMAL": float(1 - prediction)
    }

# build interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="Result"),
    title="🫁 Pneumonia Detector",
    description="""
    ## Chest X-Ray Pneumonia Detection
    Upload a chest X-ray image and the AI will detect 
    whether it shows signs of **Pneumonia** or is **Normal**.
    
    > ⚠️ This is a student project — not a medical diagnosis tool.
    """,
    examples=[],
    theme=gr.themes.Soft()
)

demo.launch()