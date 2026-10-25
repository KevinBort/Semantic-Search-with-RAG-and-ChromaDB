import asyncio
import sys
import time

from ingest import ingest
from rag import NO_ANSWER, RAGResponse, get_rag_response

# Pregunta cuya respuesta SÍ está en los documentos
REAL_QUESTION = "¿Cuánto cuesta la cuota mensual para estudiantes?"
EXPECTED_TEXT = "2.500"
EXPECTED_SOURCE = "01_reglamento_socios.md"

# Pregunta trampa: suena al tema, pero la respuesta NO está en los documentos
TRICK_QUESTION = "¿Cuál es el horario de la feria de verduras de los domingos?"


def check_real(response: RAGResponse) -> list[str]:
    errors = []
    if response.answer.strip() == NO_ANSWER:
        errors.append("respondió 'No lo sé' aunque la respuesta está en los documentos")
    if EXPECTED_TEXT not in response.answer:
        errors.append(f"la respuesta no menciona '{EXPECTED_TEXT}'")
    if EXPECTED_SOURCE not in response.references:
        errors.append(f"falta la referencia a '{EXPECTED_SOURCE}'")
    return errors


def check_trick(response: RAGResponse) -> list[str]:
    errors = []
    if response.answer.strip() != NO_ANSWER:
        errors.append(f"debía responder '{NO_ANSWER}' (posible alucinación)")
    if response.references:
        errors.append("debía devolver una lista de referencias vacía")
    return errors


def report(title: str, question: str, response: RAGResponse, errors: list[str]) -> None:
    status = "PASA" if not errors else "FALLA"
    print(f"[{status}] {title}")
    print(f"  Pregunta:    {question}")
    print(f"  Respuesta:   {response.answer}")
    print(f"  Referencias: {response.references}")
    for error in errors:
        print(f"  - {error}")
    print()


async def main() -> int:
    ingest()  # no hace nada si la colección ya existe

    start = time.perf_counter()
    # Las dos consultas corren en paralelo gracias a que get_rag_response es asíncrona
    real, trick = await asyncio.gather(
        get_rag_response(REAL_QUESTION),
        get_rag_response(TRICK_QUESTION),
    )
    elapsed = time.perf_counter() - start
    print(f"\nAmbas consultas resueltas en {elapsed:.1f} s\n")

    real_errors = check_real(real)
    trick_errors = check_trick(trick)
    report("Pregunta con respuesta en los documentos", REAL_QUESTION, real, real_errors)
    report("Pregunta trampa (no debe alucinar)", TRICK_QUESTION, trick, trick_errors)

    return 1 if real_errors or trick_errors else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
