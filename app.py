
import os
from uuid import uuid4

from session import Session

#Flask Imports
from flask import Flask
from flask import request
from flask import jsonify

#AI Imports
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate

#AWS Imports
import boto3

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

   my_sessions = {}

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

      session_id = data["sessionId"]

      print("DATA :" + str(request.data))
      print("JSON :" + str(data))
      print(" MODEL      :" + str(data["model"]))
      print(" TEMPERATURE:" + str(data["temperature"]))
      print(" TOP P      :" + str(data["topP"]))
      print(" MAX TOKENS :" + str(data["maxTokenCount"]))
      print("   WORKSPACE:" + str(data["workspace"]))
      print("    FILENAME:" + str(data["filename"]))
      print("     SESSION:" + str(session_id))

      resp = None
      if session_id in my_sessions.keys():
         print("Session exists:" + str(session_id))
         session = my_sessions[session_id]
         resp = session.send_query(data["prompt"])
      else:
         print("Creating new Session:" + str(session_id))
 
         ws_id = data["workspace"]
         faiss_idx = get_embedding_idx_by_name_and_ws(str(data["filename"]), ws_id)

         if os.path.exists(faiss_idx):
            ebeddings_db = FAISS.load_local(faiss_idx, embeddings=EMBEDDINGS, allow_dangerous_deserialization=True)
            print("Retrieved")
            #print("Embeddings DB:" + str(ebeddings_db))
            retriever = ebeddings_db.as_retriever()

            #print("Retriver:" + str(retriever))
            session = Session(session_id, retriever, data["model"], data["temperature"], data["topP"], data["maxTokenCount"])
            my_sessions[session_id] = session
   
            resp = session.send_query(data["prompt"])
            #resp = session.send_query(data["prompt"], retriever)
            #resp = send_query(llm, data["prompt"], retriever)
            print("QRESP:" + str(resp))
            print("\n")
            answer = resp['answer']
            print("RESP:" + str(answer))
            return jsonify({"answer": answer}), 200

         else:
            return jsonify({"error": "No File uploaded"}), 400

      if "answer" in resp:
         print("QRESP:" + str(resp))
         print("\n")
         answer = resp['answer']
         print("RESP:" + str(answer))
         return jsonify({"answer": answer}), 200
      else:
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

      return jsonify({"id": uuid}), 200
    

   return app

#################################################################################
#
# Loads the word file
#
#################################################################################
def process_word_file(fullpath):
   """Load Word File"""

   loader = Docx2txtLoader(fullpath)
   doc = loader.load()

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
   splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 150)
   split_docs = splitter.split_documents(document)


   print("Split:" + filename)

   # Create a vector store using the documents and the embeddings
   vector_store = FAISS.from_documents(
      split_docs,
      EMBEDDINGS,
   )

   faiss_idx = get_embedding_idx_by_name_and_ws(filename, uuid)

   vector_store.save_local(faiss_idx)

   print(vector_store.index.ntotal)



#################################################################################
#
# Gets the name of the FAISS index from the name and workspace
#
#################################################################################
def get_embedding_idx_by_name_and_ws(filename, uuid):
   return "/".join([FAISS_INDEX, filename, str(uuid)])

#################################################################################
#
# Validates doc type
#
#################################################################################
def is_doc_of_type(filename, allowed_suffixes):
    print("checking file type:" + filename)

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_suffixes


