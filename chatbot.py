from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

class Chatbot:
    def __init__(self, model_name):
        self.load_chatbot(model_name)
        self.conv_hist = []
        self.retrieve = True
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.persist_directory = './vector_db'
    
    def prepare_db(self, pdf):
        loader_pdf = PyPDFLoader(pdf)
        pages = loader_pdf.load()
        full_text = ""
        for page in pages:
            full_text += page.page_content
        full_document = [Document(metadata={"source" : "pdf_file"}, page_content=full_text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(full_document)
        self.vectordb = Chroma.from_documents(documents=chunks, embedding=self.embedding, persist_directory=self.persist_directory)

    def load_db(self, pdf):
        # self.prepare_db(pdf)
        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        print("DB is ready")

    def load_chatbot(self, model_name):
        load_dotenv(".env")
        hf_tok = os.getenv('hf_tok')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = hf_tok)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_tok)
        self.chatbot = pipeline("text-generation", model=model, tokenizer=self.tokenizer, max_new_tokens=512, do_sample=False, temperature=None, top_p=None)

    def respond(self, question):
        if self.retrieve:
            retrieved_docs = self.vectordb.similarity_search(question, k=3)
            context = []
            for d in retrieved_docs:
                context.append(d.page_content)
                context.append('\n\n')
            context = ''.join(context)
            prompt = f"""
Answer the user's QUESTION based on the following CONTEXT.
Keep your answer ground in the information provided by the CONTEXT.
The answer should be informative.
If the QUESTION is not related to the CONTEXT, say "Your question is not related to the provided document". 

----------------------------------------

CONTEXT:
{context}
----------------------------------------

QUESTION:
{question}
            """.strip()
            print(prompt, '\n')
            self.retrieve = False
        else:
            prompt = question
        self.conv_hist.append({"role" : "user", "content" : prompt})
        chatbot_output = self.chatbot(self.conv_hist, pad_token_id = self.tokenizer.eos_token_id)[0]['generated_text'][-1]
        self.conv_hist.append(chatbot_output)
        txt = chatbot_output['content']
        print(txt)
        return txt


