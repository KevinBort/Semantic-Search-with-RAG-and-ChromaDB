import asyncio
import os
import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ingest import OLLAMA_BASE_URL, get_vectorstore, ingest

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
TOP_K = 4  # cantidad de chunks que se pasan al LLM (entre 3 y 5)
NO_ANSWER = "No lo sé."


# --- Retriever ---

def get_retriever(k: int = TOP_K) -> VectorStoreRetriever:
    """Retriever sobre la colección persistente, usando el mismo modelo de embeddings que la ingesta."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


# --- Modelo de salida ---

class RAGResponse(BaseModel):
    answer: str = Field(
        description=f'Respuesta a la pregunta basada solo en el contexto, o exactamente "{NO_ANSWER}" si el contexto no la contiene.'
    )
    references: list[str] = Field(
        description="Nombres de archivo (fuente) de los fragmentos usados para responder. Lista vacía si la respuesta es 'No lo sé.'."
    )


parser = PydanticOutputParser(pydantic_object=RAGResponse)


# --- Prompt ---

SYSTEM_PROMPT = """Eres el asistente de la Cooperativa Huerta Viva. Respondes preguntas usando ÚNICAMENTE la información del CONTEXTO.

Reglas:
1. Usa solo el CONTEXTO. No uses conocimiento propio ni hagas suposiciones.
2. Si el CONTEXTO no contiene la respuesta, "answer" debe ser exactamente "{no_answer}" y "references" una lista vacía.
3. No inventes datos que no estén en el CONTEXTO (horarios, precios, nombres, fechas), aunque parezcan razonables.
4. Si la respuesta está en el CONTEXTO, responde en español, de forma breve, y en "references" pon los nombres de archivo (lo que aparece en [fuente: ...]) de los fragmentos que usaste.

{format_instructions}"""

HUMAN_PROMPT = """CONTEXTO:
{context}

PREGUNTA: {question}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
).partial(no_answer=NO_ANSWER, format_instructions=parser.get_format_instructions())


def format_docs(docs: list[Document]) -> str:
    """Une los chunks recuperados en un solo texto, indicando la fuente de cada uno."""
    return "\n\n---\n\n".join(f"[fuente: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs)


# --- Cadena LCEL ---

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,  # respuestas deterministas, menos "creatividad"
    format="json",  # obliga a Ollama a devolver JSON válido
)

rag_chain = (
    {"context": get_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)


async def get_rag_response(query: str) -> RAGResponse:
    """Busca en ChromaDB, arma el prompt, llama al LLM de forma asíncrona y parsea la respuesta."""
    return await rag_chain.ainvoke(query)


if __name__ == "__main__":
    ingest()  # no hace nada si la colección ya existe

    query = sys.argv[1] if len(sys.argv) > 1 else "¿Cuánto cuesta la cuota mensual para estudiantes?"
    response = asyncio.run(get_rag_response(query))

    print(f"\nPregunta:    {query}")
    print(f"Respuesta:   {response.answer}")
    print(f"Referencias: {response.references}")
