import os
import bs4
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader


####
import os

os.environ["LANGCHAIN_PROJECT"] = "rag-application"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""


####

# OpenAI API 키를 환경변수로 설정
os.environ["OPENAI_API_KEY"] = ""

# Only keep title, and content from the full HTML.
bs4_strainer = SoupStrainer(
    "div", attrs={"class": ["newsct_article _article_body", "media_end_head_title"]}
)
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/mnews/article/022/0003935727?sid=102",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

print(len(docs[0].page_content))

####
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=50, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

print(len(all_splits[0].page_content))

print(all_splits[2].metadata)

####
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

####

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retrieved_docs = retriever.invoke("사유지 협의매수 절차가 어떻게 되나요?")

print(len(retrieved_docs))

print(retrieved_docs[0].page_content)


####

from langchain import hub

# - https://smith.langchain.com/hub/rlm/rag-prompt
#
# You are an assistant for question-answering tasks. Use the following pieces
# of retrieved context to answer the question. If you don't know the answer, just
# say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    dict(context="context 정보", question="궁금한 질문")
).to_messages()
print(example_messages[0].content)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="o4-mini-2025-04-16")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 검색된 문서를 하나로 합쳐 줌
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# chain 생성
# RunnablePassthrough는 입력을 변경하지 않고 그대로 전달
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("사유지 협의매수 절차가 어떻게 되나요?")

for chunk in rag_chain.stream("사유지 협의매수 절차가 어떻게 되나요?"):
    print(chunk, end="", flush=True)

print(rag_chain.invoke("예산 투입 규모는 얼마나 되나요?"))

print(rag_chain.invoke("둘레길 확장 계획은 무엇인가요? bullet points로 정리해주세요"))

print(rag_chain.invoke("대구시의 둘레길 수는 얼마나 되나요?"))


####

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat history와 user input을 토대로 독립형 질문 재구성
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


####
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

####

from langchain_core.messages import HumanMessage

chat_history: list = []

# question = "둘레길 사유지 매입 절차가 어떻게 되나요?"
# ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

# second_question = "그거 하는데 돈은 얼마나 드나요?"
# ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

# print(ai_msg_2["answer"])


####

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# print(
#     conversational_rag_chain.invoke(
#         {"input": "둘레길 사유지 매입 하는 곳이 어디야?"},
#         config={
#             "configurable": {"session_id": "abc123"}
#         },  # constructs a key "abc123" in `store`.
#     )["answer"]
# )

# print(
#     conversational_rag_chain.invoke(
#         {"input": "그거 하는데 돈은 얼마나 써?"},
#         config={"configurable": {"session_id": "abc123"}},
#     )
# )

# print(
#     conversational_rag_chain.invoke(
#         {"input": "강원도는?"}, config={"configurable": {"session_id": "abc123"}}
#     )
# )

####

from langchain_community.callbacks.manager import get_openai_callback

with get_openai_callback() as cb:
    result = rag_chain.invoke(
        {"input": "충청도 착한 식당은 어디인가요?", "chat_history": []}
    )

print(cb)


####
