# recommendation_functions.py
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from collections import Counter
from statistics import mean

from helperfuncs.qdrant_functions import load_single_db


def _most_frequent_paper(sim_search_results_list):
    """
    :param sim_search_results_list: List of paper titles returned from similarity search documents
    :return: most frequent title(string)
    """
    occurence_count = Counter(sim_search_results_list)
    return occurence_count.most_common(1)[0][0]


def recommend_paper(db, query):
    """
    :param db: Qdrant db of combined embeddings
    :param query: User prompt
    :return: Paper title to recommend
    """
    # Do Similarity Search with k = 6
    sim_search_results = Qdrant.similarity_search_with_relevance_scores(db, query, k=20)

    # Get All Titles From Similarity Search
    sim_search_title_list = []

    # Get Results in List
    for res in sim_search_results:
        # Get chunk from doc
        chunk, _ = res

        # Get title meta data from chunk
        _, meta = list(chunk)[1]

        # Append title to list
        sim_search_title_list.append(meta["title"])

    return _most_frequent_paper(sim_search_title_list)


def get_list_paper_info(paper_title, meta_df):
    """
    :param paper_title: Title of recommended paper
    :param meta_df: Data frame of meta data
    :return:
    """
    # Get Index Based on Title
    index = meta_df.index[meta_df['Title'] == paper_title].to_list()[0]

    return meta_df.iloc[index]


# Mean Similarity Score
def score_recommended_paper(db, query, type_score="mean"):
    """
    :param db: Qdrant db of combined embeddings
    :param query: User prompt
    :param type_score: Keeping in case I ever want to change to top score instead of mean
    :return: Mean of cosine similarity search results for recommended paper
    """
    sim_score_list = []

    # Get Paper name
    paper_name = recommend_paper(db, query)

    # Get Sim Search Results
    sim_search = Qdrant.similarity_search_with_relevance_scores(db, query, k=6)

    # Check if Recommended Paper is in sim search document (6 checks)
    for doc in sim_search:
        doc_out, doc_meta = doc

        if doc_out.metadata["title"] == paper_name:
            sim_score_list.append(doc_meta)

    if type_score == "top":
        return sim_score_list[0]

    return mean(sim_score_list)


def get_summary_recommended_paper(client, pmid):
    """
    :param client: Qdrant client
    :param pmid: Pubmed ID of paper
    :return: Summary of paper
    """
    paper_db = load_single_db(client, pmid)

    summary_query = "Please summarize the main findings and key experiments done in this paper"
    sim_search_docs = Qdrant.similarity_search(paper_db, summary_query, k=4)
    chain = load_qa_chain(OpenAI(model="gpt-3.5-turbo-instruct"), chain_type="stuff")

    return chain.run(input_documents=sim_search_docs, question=summary_query)


def get_best_paper_and_summary(db, query, meta_df, client):
    """
    :param db: Qdrant db from combined embeddings
    :param query: User prompt
    :param meta_df: Data frame of metadata
    :param client: Qdrant client
    :return: Dictionary of responses to use
    """
    # Get Recommended Paper Title
    recommended_paper = recommend_paper(db, query)

    # Get Paper Meta Data
    list_of_paper_meta_data = get_list_paper_info(recommended_paper, meta_df)
    pmid = list_of_paper_meta_data[1]

    # Get Paper Summary
    paper_summary = get_summary_recommended_paper(client, pmid)

    # Get Mean Similarity Score
    mean_score = score_recommended_paper(db, query)

    if mean_score > 0.65:
        confidence = "High"
    elif mean_score > 0.6:
        confidence = "Medium"
    elif mean_score > 0.5:
        confidence = "Low"
    else:
        confidence = "Extremely Low"

    # Return info dictionary
    return {"Title": list_of_paper_meta_data[3],
            "Author": list_of_paper_meta_data[0],
            "URL": list_of_paper_meta_data[2],
            "PMID": pmid,
            "Confidence": f"{confidence}: {round(mean_score,3)}",
            "Summary": paper_summary}
