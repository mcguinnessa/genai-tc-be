
#from abc import ABC, abstractmethod

from langchain_aws import ChatBedrock


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain



TEMPERATURE_IDX = 0
MAX_TOKENS_IDX  = 1
TOP_P_IDX       = 2


MODEL_DETAILS = {"amazon.titan-text-express-v1": ["temperature", "maxTokenCount", "topP"], 
                 "mistral.mistral-large-2402-v1:0": ["temperature", "max_tokens", "top_p"],
                 "mistral.mixtral-8x7b-instruct-v0:1": ["temperature", "max_tokens", "top_p"] }



TC_DEFINITION = """A Test Case is defined as having the following fields:

Number: abbreviated to No., and is the number of the test case in the table or document. This should start from one.
Test Name: A useful short description of the test case.
Description: A summary of the test case.
ID: An alphanumeric identifier, unique to the test and derived from the element under test and the number field. Must be a maximum of 7 characters. Can include _ or - characters.
Pre-Conditions: Describes the preconditions needed for the tests to be executed.
Steps: This is a series of at least 3 steps, more if needed, that clearly describes how to execute the test case. Each step shall be numbered. The steps should include the following: Any parameters that need to be set or changed and what protocols will be used; any user interfaces the tester must use in order to carry out the instructions in the test step; any connectivity actions involved in the step.
Expected Results: This describes the expected outcomes for each of the steps itemised in Test Steps, including any values that can be validated or any expected errors that will occur."""




#class Session(ABC):
class Session():

   CONTEXTUALISE_SYSTEM_PROMPT = (
      "Given a chat history and the latest user question "
      "which might reference context in the chat history, "
      "formulate a standalone question which can be understood "
      "without the chat history. Do NOT answer the question, "
      "just reformulate it if needed and otherwise return it as is."
   )

   CONTEXTUALISE_Q_PROMPT = ChatPromptTemplate.from_messages(
      [
         ("system", CONTEXTUALISE_SYSTEM_PROMPT),
         MessagesPlaceholder("chat_history"),
         ("human", "{input}"),
      ]
   )

   PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
      <context>
      {context}
      </context>"""+
      TC_DEFINITION + """
      Question: {input}""")


   def __init__(self, session_id, retriever, model_id, temperature, top_p, max_token_count):
      """Session"""

      self.session_id = session_id
      self.model_id = model_id
      self.temperature = temperature
      self.top_p = top_p
      self.max_token_count = max_token_count
      self.retriever = retriever

      self.llm = self.get_llm(self.model_id, self.temperature, self.top_p, self.max_token_count)
      self.store = {}
      self.history_chain = create_stuff_documents_chain(self.llm, self.PROMPT_TEMPLATE)

      self.history_aware_retriever = create_history_aware_retriever(
         llm=self.llm,
         retriever=self.retriever,
         prompt=self.CONTEXTUALISE_Q_PROMPT
      )

      self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.history_chain)
      print("Chain:" + str(self.rag_chain))

      self.conversational_rag_chain = RunnableWithMessageHistory(
         self.rag_chain,
         self.get_session_history,
         input_messages_key="input",
         history_messages_key="chat_history",
         output_messages_key="answer",
      )


   #################################################################################
   #
   # Gets the LLM
   #
   #################################################################################
   def get_llm(self, model, temperature, top_p, max_token_count):

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

   #################################################################################
   #
   # Splits the file and stores in the vector store in an encoded format
   #
   #################################################################################
   def send_query(self, input_text):

      resp = self.conversational_rag_chain.invoke( {"input": input_text}, config={ "configurable": {"session_id": self.session_id} })
      print("Resp:" + str(resp))
      return resp


   #################################################################################
   #
   # Gets the history of the session
   #
   #################################################################################
   def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
       if session_id not in self.store:
           self.store[session_id] = ChatMessageHistory()
       return self.store[session_id]


