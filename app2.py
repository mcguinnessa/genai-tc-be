
import os
from uuid import uuid4

#Flask Imports
from flask import Flask
from flask import request
from flask import jsonify

#AI Imports
from langchain_aws import ChatBedrock
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


#AWS Imports
import boto3

TEMPERATURE_IDX = 0
MAX_TOKENS_IDX  = 1
TOP_P_IDX       = 2
MODEL_DETAILS = {"amazon.titan-text-express-v1": ["temperature", "maxTokenCount", "topP"], 
                 "mistral.mixtral-8x7b-instruct-v0:1": ["temperature", "max_tokens", "top_p"] }

TC_DEFINITION = """A Test Case is defined as having the following fields:

Number: abbreviated to No., and is the number of the test case in the table or document. This should start from one.
Test Name: A useful short description of the test case.
Description: A summary of the test case.
ID: An alphanumeric identifier, unique to the test and derived from the element under test and the number field. Must be a maximum of 7 characters. Can include _ or - characters.
Pre-Conditions: Describes the preconditions needed for the tests to be executed.
Steps: This is a series of at least 3 steps, more if needed, that clearly describes how to execute the test case. Each step shall be numbered. The steps should include the following: Any parameters that need to be set or changed and what protocols will be used; any user interfaces the tester must use in order to carry out the instructions in the test step; any connectivity actions involved in the step.
Expected Results: This describes the expected outcomes for each of the steps itemised in Test Steps, including any values that can be validated or any expected errors that will occur."""


EMBEDDINGS_MODEL = "amazon.titan-embed-text-v2:0"
FAISS_INDEX = "faiss_db_index"

BEDROCK_RT = boto3.client(service_name='bedrock-runtime')
EMBEDDINGS = BedrockEmbeddings(model_id=EMBEDDINGS_MODEL, client=BEDROCK_RT)

ALLOWED_WORD_EXTENSIONS = {'docx'}
ALLOWED_PFD_EXTENSIONS = {'pdf'}

def create_app(test_config=None):
   """Create the App"""
   print("Creating the App")
   app = Flask(__name__)

   @app.route("/")
   def hello_world():
      return "<p>Hello, World!</p>"

   #################################################################################
   #
   # Responds to the heath check
   #
   #################################################################################
   @app.route("/health")
   def health():
      return "<p>OK</p>"

   #################################################################################
   #
   # Returns a list of the workspaces
   #
   #################################################################################
   @app.get("/workspaces")
   def list_workspaces():
      print("This will list the workspaces")
      #faiss_idx = "/".join([FAISS_INDEX, uuid])

      rc = {}
      workspaces = []

      if os.path.exists(FAISS_INDEX):
 
         for fn in os.listdir(FAISS_INDEX):
            print("Files:" + str(fn))
            ws = {}
            path = os.path.join(FAISS_INDEX, fn)
            if os.path.isdir(path):
               for id in os.listdir(path):
                  print("Folder:" + str(id)) 
                  ws["filename"] = fn
                  ws["id"] = id
               workspaces.append(ws)
      else:
         print(FAISS_INDEX + " does not exist")

      rc["workspaces"] = workspaces
        
      return jsonify(rc), 200
          

      #files = [f for f in os.listdir(FAISS_INDEX) if os.path.isdir(os.path.join(FAISS_INDEX, f))]
      #print("Files:" + str(files))

   #################################################################################
   #
   # Generate the test cases
   #
   #################################################################################
   @app.post("/generate")
   def generate():
      print("This is a query")

      data = request.json

      print("DATA :" + str(request.data))
      print("JSON :" + str(data))
      print(" MODEL      :" + str(data["model"]))
      print(" TEMPERATURE:" + str(data["temperature"]))
      print(" TOP P      :" + str(data["topP"]))
      print(" MAX TOKENS :" + str(data["maxTokenCount"]))
      print("   WORKSPACE:" + str(data["workspace"]))
      print("    FILENAME:" + str(data["filename"]))

 
     client = boto3.client('bedrock', region_name='eu-west-2')  # Replace with your AWS region
     embedding_model = BedrockEmbeddings(client=client)
     llm = Bedrock(client=client, model_id='bedrock-model-id')  # Replace with your model ID

     vector_store = FAISS(embedding_model)



   #################################################################################
   #
   # Generate the test cases
   #
   #################################################################################
   @app.post("/generate2")
   def generate():
      print("This is a query2")

      data = request.json

      print("DATA :" + str(request.data))
      print("JSON :" + str(data))
      print(" MODEL      :" + str(data["model"]))
      print(" TEMPERATURE:" + str(data["temperature"]))
      print(" TOP P      :" + str(data["topP"]))
      print(" MAX TOKENS :" + str(data["maxTokenCount"]))
      print("   WORKSPACE:" + str(data["workspace"]))
      print("    FILENAME:" + str(data["filename"]))

      llm = get_llm(data["model"], data["temperature"], data["topP"], data["maxTokenCount"])

      #uuid = "12321-23423-24234"
      uuid = data["workspace"]
      #faiss_idx = "/".join([FAISS_INDEX, uuid])
      faiss_idx = get_embedding_by_name_and_ws(str(data["filename"]), uuid)

      #if os.path.exists(FAISS_INDEX):
      if os.path.exists(faiss_idx):
         ebeddings_db = FAISS.load_local(faiss_idx, embeddings=EMBEDDINGS, allow_dangerous_deserialization=True)
         print("Retrieved")
         #print("Embeddings DB:" + str(ebeddings_db))
         retriever = ebeddings_db.as_retriever()
         #print("Retriver:" + str(retriever))

         #memory = create_memory(llm)
         
         resp = send_query(llm, data["prompt"], retriever)
         #resp = send_query(llm, "What day is it?", retriever)
         print("QRESP:" + str(resp))
         print("\n")
         answer = resp['answer']
         print("RESP:" + str(answer))
         return jsonify({"answer": answer}), 200

      else:
         return jsonify({"error": "No File uploaded"}), 400

      return jsonify({"error": "No Data generated"}), 400

   #################################################################################
   #
   # Receives the file
   #
   #################################################################################
   @app.post('/upload')
   def upload_file():
      print("This is an upload")
      print("File:" + str(request.files))

      uuid = uuid4()
      print("UUID:" + str(uuid))

      ################## USED FOR TESTING TO SAVE AWS CALLS
#      return jsonify({"id": uuid}), 200

      if 'file' not in request.files:
         return jsonify({"error": "No file part"}), 400
    
      f = request.files['file']
    
      if f.filename == '':
         return jsonify({"error": "No selected file"}), 400

      temp_file_path = os.path.join('/tmp', f.filename + ".tmp")
      f.save(temp_file_path)
   
      document = None 
      if f and is_doc_of_type(f.filename, ALLOWED_WORD_EXTENSIONS):
         print("File is Word Doc!")
         document = process_word_file(temp_file_path)
      else:
         return jsonify({"error": "Invalid file type"}), 400

      # Clean up the temporary file
      #os.remove(temp_file_path)

      split_encode_and_store_file(document, f.filename, uuid)

      #return jsonify({"content": [doc.page_content for doc in document]}), 200
      return jsonify({"id": uuid}), 200
    

   return app

#################################################################################
#
# Gets the LLM
#
#################################################################################
def get_llm(model, temperature, top_p, max_token_count):

   print(" MODEL      :" + str(model))
   print(" TEMPERATURE:" + str(temperature))
   print(" TOP P      :" + str(top_p))
   print(" MAX TOKENS :" + str(max_token_count))
         
   detail_names = MODEL_DETAILS[model]

   return ChatBedrock (
      model_id = model,
      model_kwargs={
         detail_names[TEMPERATURE_IDX]: temperature,
         detail_names[MAX_TOKENS_IDX]: max_token_count,
         detail_names[TOP_P_IDX]: top_p
      }
   )


#   return ChatBedrock(
#      #model_id ="amazon.titan-text-express-v1",
#      model_id = model,
#      model_kwargs={
#         "temperature":temperature,
#         "maxTokenCount": max_token_count,
#         "topP": top_p
#      }
#   )

#   return ChatBedrock(
#      model_id = model,
#      model_kwargs={
#         "temperature":temperature,
#         "max_tokens": max_token_count,
#         "top_p": top_p
#      }
#   )

#################################################################################
#
# Loads the word file
#
#################################################################################
def process_word_file(fullpath):
   """Load Word File"""

   loader = Docx2txtLoader(fullpath)
   doc = loader.load()

#   return Docx2txtLoader(fullpath).load()
   return doc

#################################################################################
#
# Splits the file and stores in the vector store in an encoded format
#
#################################################################################
def split_encode_and_store_file(document, filename, uuid):
   """Load Word File"""
   print("Spliting:" + filename)

   #split into chunks of 1500 characters each, with an overlap of 150 characters between adjacent chunks.
   splitter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap = 150)
   split_docs = splitter.create_documents([datum.page_content for datum in document])
   print("Split:" + filename)

   # Create a vector store using the documents and the embeddings
   vector_store = FAISS.from_documents(
      split_docs,
      EMBEDDINGS,
   )
   # Save the vector store locally

#   faiss_idx = "/".join([FAISS_INDEX, filename, str(uuid)])
#   def get_embedding_by_name_and_ws(filename, uuid):
   faiss_idx = get_embedding_by_name_and_ws(filename, uuid)

   #vector_store.save_local(FAISS_INDEX)
   vector_store.save_local(faiss_idx)

   print(vector_store.index.ntotal)


#################################################################################
#
# Gets the name of the FAISS index from the name and workspace
#
#################################################################################
def get_embedding_by_name_and_ws(filename, uuid):
   return "/".join([FAISS_INDEX, filename, str(uuid)])

#################################################################################
#
# Splits the file and stores in the vector store in an encoded format
#
#################################################################################
def send_query(llm, input_text, retriever):

   prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
      <context>
      {context}
      </context>"""+
      TC_DEFINITION + """
      Question: {input}""")

   document_chain = create_stuff_documents_chain(llm, prompt)

   retrieval_chain = create_retrieval_chain(retriever, document_chain)
   print("Chain:" + str(retrieval_chain))
   return retrieval_chain.invoke({"input": input_text})

#################################################################################
#
# Creates the memory for context
#
#################################################################################
#def create_memory(llm):
#   #keeps summary of previous messages, max token limit forces flushing of old data
#   return ConversationSummaryBufferMemory(llm=llm, max_token_limit=256) 

#################################################################################
#
# Checks the file is a word file
#
#################################################################################
#def is_word_doc(filename):
#    print("checking file type:" + filename)
#    ALLOWED_EXTENSIONS = {'docx'}
#
#    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_doc_of_type(filename, allowed_suffixes):
    print("checking file type:" + filename)
    #ALLOWED_EXTENSIONS = {'docx'}

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_suffixes


