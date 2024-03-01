import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

HEIGHT = 128
WIDTH = 128

@st.cache(allow_output_mutation=True)  # Modified cache decorator
def load_model():
    model_path = "ResNEt50.h5"  # Corrected model path
    return tf.keras.models.load_model(model_path)

def main():
    st.title("Retinopathy Eye Detection using Deep Learning")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Predict"):
            predicted_class, lime_explanation = predict(image)
            st.write('Predicted Class:', predicted_class)
            st.image(lime_explanation.image, caption='Lime Explanation', use_column_width=True)

def predict(image):
    batch_size = 1
    new_model = load_model()
    
    # Prepare the image
    image_resized = np.array(image.resize((WIDTH, HEIGHT))) / 255.0
    image_expanded = np.expand_dims(image_resized, axis=0)
    
    # Get the predicted class
    y_pred = new_model.predict(image_expanded, batch_size=batch_size, verbose=0)
    predicted_class = np.argmax(y_pred)
    
    # Lime explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_resized,  # Removed 'astype' as it's already in the correct format
                                             new_model.predict, 
                                             top_labels=1, 
                                             hide_color=0, 
                                             num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    
    # Mark boundaries
    marked_image = mark_boundaries(temp / 2 + 0.5, mask)
    
    lime_explanation = lime_image.ImageExplanation(marked_image, mask)
    
    return predicted_class, lime_explanation

if __name__ == '__main__':
    main()
