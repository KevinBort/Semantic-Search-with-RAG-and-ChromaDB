# Semantic Search with RAG and ChromaDB

Sistema de recuperación semántica local (RAG) hecho con **LangChain (LCEL)**, **ChromaDB** y **Ollama**. Responde preguntas usando **únicamente** la información de los documentos de `/data`; si la respuesta no está ahí, responde `"No lo sé."`.

Todo corre en tu computadora: no hace falta ninguna API paga ni API key.

## Arquitectura

```
                 INGESTA (ingest.py, se ejecuta una sola vez)
 data/*.md ──► carga ──► chunking (500 tokens, overlap 50) ──► embeddings (bge-m3) ──► ChromaDB (./vectorstore)

                 CONSULTA (rag.py)
 pregunta ──► embedding (bge-m3) ──► ChromaDB top_k=4 ──► format_docs ──► prompt ──► LLM (qwen3) ──► PydanticOutputParser ──► RAGResponse
```

| Componente | Herramienta |
|---|---|
| LLM | `qwen3:4b-instruct` vía Ollama |
| Embeddings | `bge-m3` vía Ollama (multilingüe, 1024 dimensiones) |
| Base vectorial | ChromaDB persistente en `./vectorstore` |
| Orquestación | LangChain con LCEL |
| Salida estructurada | Pydantic + `PydanticOutputParser` |

## Estructura del repositorio

```
.
├── data/                      # Dataset de ejemplo (4 archivos .md)
│   ├── 01_reglamento_socios.md
│   ├── 02_guia_cultivo.md
│   ├── 03_compostaje.md
│   └── 04_talleres_y_faq.md
├── ingest.py                  # Módulo de ingesta: carga, chunking y persistencia en ChromaDB
├── rag.py                     # Retriever, prompt, cadena LCEL y get_rag_response() asíncrona
├── test_rag.py                # Pruebas: pregunta real y pregunta trampa
├── requirements.txt
├── .env.example               # Variables de entorno de ejemplo
└── README.md
```

## Dataset

Los documentos describen una cooperativa de huertas urbanas **ficticia**, "Huerta Viva": reglamento de socios, guía de cultivo, manual de compostaje y talleres/FAQ.

Se eligió un tema inventado a propósito: como esos datos (cuotas, horarios, nombres) no existen en ningún otro lado, el LLM no puede conocerlos de antemano. Si responde bien, es porque los sacó del contexto recuperado, y si inventa algo, se nota enseguida.

## Instalación

### 1. Ollama y los modelos

Instalá Ollama desde [ollama.com](https://ollama.com) y descargá los dos modelos:

```bash
ollama pull qwen3:4b-instruct
ollama pull bge-m3
```

### 2. Entorno virtual y dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Variables de entorno

```bash
cp .env.example .env
```

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Dirección del servidor de Ollama |
| `LLM_MODEL` | `qwen3:4b-instruct` | Modelo que genera las respuestas |
| `EMBEDDING_MODEL` | `bge-m3` | Modelo de embeddings (indexación y consulta) |
| `CHROMA_DIR` | `./vectorstore` | Carpeta donde se persiste ChromaDB |
| `COLLECTION_NAME` | `rag_docs` | Nombre de la colección |

## Uso

### Ingesta

```bash
python ingest.py
```

La primera vez genera los embeddings y los guarda en `./vectorstore`. En las siguientes ejecuciones detecta que la colección ya tiene datos y **no re-indexa**.

Si cambiás los documentos de `/data` o los parámetros de chunking, forzá la re-indexación con:

```bash
python ingest.py --reset
```

### Hacer una pregunta

```bash
python rag.py "¿Qué preparado se usa contra los pulgones?"
```

```
Pregunta:    ¿Qué preparado se usa contra los pulgones?
Respuesta:   El Purín de ortiga se usa contra los pulgones.
Referencias: ['02_guia_cultivo.md']
```

`rag.py` llama a `ingest()` al arrancar, así que también funciona aunque no se haya ejecutado la ingesta antes.

### Desde código

```python
import asyncio
from rag import get_rag_response

response = asyncio.run(get_rag_response("¿Cuánto cuesta la cuota mensual?"))
print(response.answer)       # texto de la respuesta
print(response.references)   # archivos usados como fuente
```

## Pruebas

```bash
python test_rag.py
```

Ejecuta dos consultas **en paralelo** con `asyncio.gather` y verifica automáticamente los resultados:

| Prueba | Pregunta | Resultado esperado |
|---|---|---|
| Respuesta en los documentos | ¿Cuánto cuesta la cuota mensual para estudiantes? | Menciona "2.500" y cita `01_reglamento_socios.md` |
| Pregunta trampa | ¿Cuál es el horario de la feria de verduras de los domingos? | Exactamente `"No lo sé."` y sin referencias |

Resultado obtenido:

```
[PASA] Pregunta con respuesta en los documentos
  Pregunta:    ¿Cuánto cuesta la cuota mensual para estudiantes?
  Respuesta:   2.500 pesos
  Referencias: ['01_reglamento_socios.md']

[PASA] Pregunta trampa (no debe alucinar)
  Pregunta:    ¿Cuál es el horario de la feria de verduras de los domingos?
  Respuesta:   No lo sé.
  Referencias: []
```

La pregunta trampa es deliberadamente cercana al tema: los documentos mencionan varios horarios de los sábados, pero ninguna feria de los domingos. Es un caso donde un modelo sin un prompt restrictivo tiende a inventar un horario.

El script devuelve código de salida `0` si ambas pruebas pasan y `1` si alguna falla.

## Decisiones de diseño

### Ollama en lugar de una API paga

Los modelos corren localmente: no hay costos, no hay límites de uso y los documentos nunca salen de la máquina. No hay API keys que proteger; igual se usa `.env` para la configuración, como pide la consigna.

### Chunking

Se usa `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=500` y `chunk_overlap=50`, medidos en **tokens** (no en caracteres). El splitter intenta cortar primero entre párrafos (`\n\n`), luego entre líneas, luego entre palabras, así que cada chunk suele coincidir con una sección completa del documento. Los 4 documentos generan 12 chunks.

El overlap real varía entre 0 y 50 tokens: el splitter solo repite párrafos completos que entren en esos 50 tokens y nunca corta un párrafo por la mitad para generar solapamiento.

### Mismo modelo de embeddings para indexar y consultar

El modelo de embeddings se define en una sola función, `get_embeddings()` en `ingest.py`, que usan tanto la ingesta como el retriever. Así es imposible indexar con un modelo y consultar con otro.

### Persistencia

Antes de indexar, `ingest()` consulta si la colección ya tiene al menos un registro. Esto es más confiable que chequear si la carpeta existe, porque la carpeta puede existir vacía si una ingesta anterior falló.

### `top_k = 4`

Se recuperan 4 chunks por consulta, dentro del rango de 3 a 5 recomendado, para no saturar al modelo de contexto (*lost in the middle*).

El retriever **siempre** devuelve 4 chunks, aunque ninguno sea relevante. Por eso la decisión de responder "No lo sé" queda a cargo del prompt y no del retriever.

### Prompt como "filtro de veracidad"

El prompt de sistema le indica al modelo que:

1. Use solo el CONTEXTO, sin conocimiento propio.
2. Si la respuesta no está, responda exactamente `"No lo sé."` con referencias vacías.
3. No invente horarios, precios, nombres ni fechas, aunque parezcan razonables.
4. Cite en `references` los archivos que usó; cada chunk llega etiquetado como `[fuente: archivo.md]`.

### Salida estructurada

La respuesta se parsea a este modelo Pydantic:

```python
class RAGResponse(BaseModel):
    answer: str
    references: list[str]
```

Además, el LLM se configura con `temperature=0`, para respuestas deterministas, y `format="json"`, para que Ollama solo pueda generar JSON válido y el parser no falle por texto extra.

### Cadena LCEL

```python
rag_chain = (
    {"context": get_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

`get_rag_response(query)` ejecuta la cadena con `ainvoke`, de forma asíncrona.
