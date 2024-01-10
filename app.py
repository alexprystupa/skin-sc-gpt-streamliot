# app.py
import s3fs
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

from helperfuncs.aws_functions import get_aws_S3_client
from helperfuncs.aws_functions import get_s3_first_img_path_all_papers
from helperfuncs.aws_functions import get_s3_img_paths_recommended_paper
from helperfuncs.aws_functions import read_S3_meta_data
from helperfuncs.pdf_chat_functions import get_conversation_chain
from helperfuncs.qdrant_functions import load_qdrant_client
from helperfuncs.qdrant_functions import load_single_db
from helperfuncs.recommendation_functions import get_best_paper_and_summary

# Load Qdrant Client
client = load_qdrant_client(url=st.secrets["QDRANT_HOST"], api_key=st.secrets["QDRANT_API_KEY"])

# Load AWS Client
s3_client = get_aws_S3_client(access_key=st.secrets["AWS_ACCESS_KEY"], secret_key=st.secrets["AWS_SECRET_KEY"])
fs = s3fs.S3FileSystem(anon=False, key=st.secrets["AWS_ACCESS_KEY"], secret=st.secrets["AWS_SECRET_KEY"])

# Global Variables
meta_df = read_S3_meta_data(s3_client)
db = Qdrant(client=client, collection_name="combined_pdf_docs", embeddings=HuggingFaceEmbeddings())
pdf_img_path = "/Users/prysta01/Desktop/Skin_Sequencing_GPT_Project/streamlit-testing/data/pdf-images"


def set_chat_session_state(prompt):
    st.session_state["prompt"] = prompt


@st.cache_data
def get_list_first_page_all_papers():
    s3_img_paths = get_s3_first_img_path_all_papers(s3_client=s3_client, bucket_name="sc-pdf-recommendation-bucket")

    return [fs.open(s3_img_path, mode='rb').read() for s3_img_path in s3_img_paths]

    
@st.cache_data
def get_list_recommended_paper_imgs(pmid):

    s3_img_paths = get_s3_img_paths_recommended_paper(s3_client=s3_client, bucket_name="sc-pdf-recommendation-bucket",
                                                      pmid=str(pmid))

    return [fs.open(s3_img_path, mode='rb').read() for s3_img_path in s3_img_paths]


def main():

    # Set Session States
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Set Page Layouts
    st.set_page_config(page_title="SC-PDF", page_icon=":books:")

    if st.session_state["prompt"]:
        st.header("Chat with recommended paper :male-scientist:")

        with st.sidebar:
            st.sidebar.title(st.session_state['prompt_response_dict']['Title'])
            st.write("-----")

            paper_images = get_list_recommended_paper_imgs(st.session_state['prompt_response_dict']['PMID'])

            for paper_img in paper_images:
                st.image(paper_img)

    else:
        st.header("Single Cell PDF Recommendation System :male-scientist:")

    prompt = st.chat_input("Say something")

    # No Prompt Has been asked by the user
    if not prompt and not st.session_state["prompt"]:
        with st.sidebar:
            st.sidebar.title("Single Cell Skin Sequencing Papers")
            st.write("-----")

            first_page_images = get_list_first_page_all_papers()

            for i, first_page in enumerate(first_page_images):
                st.header(f"Paper {i + 1}")
                st.image(first_page)

    # Prompt has been asked by the user showing recommended paper
    if prompt and not st.session_state["prompt"]:
        prompt_response_dict = get_best_paper_and_summary(db, prompt, meta_df, client)
        st.session_state["prompt_response_dict"] = prompt_response_dict
        with st.chat_message("user"):
            st.header(prompt_response_dict["Title"], divider='rainbow')
            st.subheader(prompt_response_dict["Author"])
            st.subheader(prompt_response_dict["URL"])
            st.subheader(f"Response Confidence: {prompt_response_dict['Confidence']}", divider='rainbow')
            st.subheader("Summary")
            st.write(prompt_response_dict["Summary"])

        with st.sidebar:
            st.sidebar.title(prompt_response_dict["Title"])
            st.write("-----")

            st.button("Chat with pdf", on_click=set_chat_session_state, args=(prompt,))

            paper_images = get_list_recommended_paper_imgs(st.session_state['prompt_response_dict']['PMID'])

            for paper_img in paper_images:
                st.image(paper_img)

    # User has chosen to ask questions to the paper
    if prompt and st.session_state["prompt"]:
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            single_db = load_single_db(client, st.session_state['prompt_response_dict']['PMID'])
            st.session_state.conversation = get_conversation_chain(single_db)

        st.session_state.chat_history.append(st.session_state.conversation({"question": prompt}))

        for chat in st.session_state.chat_history:
            # Write user prompt
            with st.chat_message("user"):
                st.write(chat["question"])

            # Write the question response
            with st.chat_message("assistant"):
                st.write(chat["answer"])


if __name__ == '__main__':
    main()
