import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings


def load_qdrant_client(url, api_key):
    """
    :param url: URL located in secrets.toml file
    :param api_key: API key located in secrets.toml file
    :return: Connection to Qdrant Client Cloud
    """
    return qdrant_client.QdrantClient(url=url, api_key=api_key)


def load_single_db(client, pmid):
    """
    :param client: Qdrant Client
    :param pmid: Pubmed ID of paper
    :return: Qdrant DB of specific paper
    """
    return Qdrant(client=client, collection_name=f"PDF-{pmid}", embeddings=HuggingFaceEmbeddings())
