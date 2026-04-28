import streamlit as st
import promptinjection
import os

if 'app' not in st.session_state:
    st.session_state.app = promptinjection.build_graph()

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("CV reviewer chatbot")

api_key = st.sidebar.text_input("Enter your GROQ API key", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("GROQ_API_KEY has been set.")

selected_model = st.sidebar.selectbox(
    "Groq model",
    promptinjection.EVALUATION_MODELS,
    index=0,
)
use_sanitization = st.sidebar.checkbox("Enable data sanitization", value=True)
use_guardrails = st.sidebar.checkbox("Enable semantic output guardrails", value=True)

UPLOAD_FOLDER = str(promptinjection.CVS_DIR)

pdf_file = st.sidebar.file_uploader("Upload a PDF CV", type=["pdf"])

if pdf_file:

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, pdf_file.name)

    with open(file_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    st.sidebar.success(f"CV successfully uploaded")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.sidebar.subheader("Baseline model evaluation")
evaluation_prompt = st.sidebar.text_area(
    "Evaluation prompt",
    value=promptinjection.DEFAULT_EVAL_PROMPT,
    height=100,
)

if st.sidebar.button("Run model comparison"):
    if not os.environ.get("GROQ_API_KEY"):
        st.sidebar.error("Add a GROQ API key before running model evaluation.")
    else:
        with st.spinner("Running baseline comparison across models..."):
            results = promptinjection.evaluate_models(
                prompt=evaluation_prompt,
                models=promptinjection.EVALUATION_MODELS,
                use_sanitization=False,
                use_guardrails=False,
            )
        st.subheader("Baseline comparison results")
        for result in results:
            st.markdown(f"**{result['model']}**")
            st.markdown(f"Verdict: {result['verdict']}")
            st.markdown(result["answer"])

if prompt := st.chat_input("Enter prompt here.."):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Add a GROQ API key in the sidebar before running the reviewer.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = st.session_state.app.invoke(
        {
            "prompt": prompt,
            "model": selected_model,
            "use_sanitization": use_sanitization,
            "use_guardrails": use_guardrails,
        }
    )
    assistant_reply = result.get("final_answer", "No response generated.")

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
