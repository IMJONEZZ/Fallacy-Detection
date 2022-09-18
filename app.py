import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "google/t5-v1_1-base"

st.header("Detect Fallacies")

st_model_load = st.text("Loading Fallacy Detector Model")

@st.cache(allow_output_mutation=True)
def load_model():
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    model = T5ForConditionalGeneration.from_pretrained("./FallacyDetectorT5")
    print("Model Loaded!")
    return tokenizer, model

tokenizer, model = load_model()
st.success("Model Loaded!")
st_model_load.text("")

if 'text' not in st.session_state:
    st.session_state.text = ""
st_text_area = st.text_area('Text to detect fallacies', value=st.session_state.text, height=500)

def detect():
    st.session_state.text = st_text_area

    inputs = ["Detect Fallacy: " + st_text_area]
    inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

    outputs = model.generate(**inputs)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    st.session_state.fallacies = decoded_outputs

st_detect_button = st.button('Detect', on_click=detect)

if 'fallacies' not in st.session_state:
    st.session_state.fallacies = []

if len(st.session_state.fallacies) > 0:
    with st.container():
        st.subheader("Detected Fallacies")
        for fallacy in st.session_state.fallacies:
            st.markdown("__" + fallacy + "__")