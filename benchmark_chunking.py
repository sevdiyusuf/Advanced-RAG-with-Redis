import os
import json
import time
import csv
from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#expected: "present" = PDF’de olmasını bekliyorsun
#expected: "absent" = PDF’de yok; model “Bu bilgi dokümanda yok.” demeli

PDF_PATH = "data/doc.pdf"
QUESTIONS_PATH = "questions.json"
OUTPUT_CSV = "results.csv"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"      # hızlı istersen: "phi3:mini"
TEMPERATURE = 0

# Her ayarda kaç parça çekilecek (retrieval)
TOP_K = 4

# Modelin “PDF’de yoksa” kullanması gereken cümle
NO_ANSWER_TEXT = "Bu bilgi dokümanda yok."


@dataclass
class ChunkConfig:
    name: str
    chunk_size: int
    chunk_overlap: int


CONFIGS = [
    ChunkConfig(name="A_500_80", chunk_size=500, chunk_overlap=80),
    ChunkConfig(name="B_700_120", chunk_size=700, chunk_overlap=120),
    ChunkConfig(name="C_1000_150", chunk_size=1000, chunk_overlap=150),
]


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Sen bir doküman asistanısın. SADECE aşağıdaki CONTEXT içindeki bilgilere dayanarak cevap ver.\n"
        f"Eğer cevap CONTEXT içinde yoksa, tam olarak şunu yaz: '{NO_ANSWER_TEXT}'\n"
        "Cevabı mümkün olduğunca kısa ve net yaz.\n\n"
        "CONTEXT:\n{context}\n\n"
        "SORU:\n{question}\n\n"
        "CEVAP:"
    ),
)


def load_questions(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} bulunamadı. questions.json oluşturmalısın.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # minimal doğrulama
    for item in data:
        if "id" not in item or "question" not in item or "expected" not in item:
            raise ValueError("questions.json her item için id/question/expected içermeli.")
        if item["expected"] not in ("present", "absent"):
            raise ValueError("expected sadece 'present' veya 'absent' olmalı.")
    return data


def build_vectorstore(pdf_path: str, cfg: ChunkConfig) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def build_qa(vs: FAISS) -> RetrievalQA:
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa


def looks_like_no_answer(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return NO_ANSWER_TEXT.lower() in a


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("data/doc.pdf bulunamadı. PDF’yi data/ içine koy ve adı doc.pdf olsun.")

    questions = load_questions(QUESTIONS_PATH)

    rows = []
    print("\n=== Chunk Tuning Benchmark Başladı ===\n")
    print(f"PDF: {PDF_PATH}")
    print(f"LLM: {LLM_MODEL} | Embedding: {EMBED_MODEL} | TOP_K: {TOP_K}\n")

    for cfg in CONFIGS:
        print(f"\n--- CONFIG: {cfg.name} (chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap}) ---")

        t0 = time.time()
        vs = build_vectorstore(PDF_PATH, cfg)
        build_time = time.time() - t0
        print(f"Index build süresi: {build_time:.2f}s")

        qa = build_qa(vs)

        for q in questions:
            qid = q["id"]
            question = q["question"]
            expected = q["expected"]  # present/absent

            t1 = time.time()
            result = qa.invoke({"query": question})
            latency = time.time() - t1

            answer = result.get("result", "")
            sources = result.get("source_documents", []) or []

            predicted_no = looks_like_no_answer(answer)
            expected_no = (expected == "absent")

            correct_no_answer_behavior = (predicted_no == expected_no)

            # kaynak sayfaları
            pages = []
            for d in sources:
                p = d.metadata.get("page", None)
                if p is not None:
                    pages.append(str(p))
            # benzersiz
            pages = sorted(set(pages), key=lambda x: int(x) if x.isdigit() else 10**9)

            print(f"  [{qid}] {latency:.2f}s | expected={expected} | no_answer_ok={correct_no_answer_behavior} | pages={pages}")

            rows.append({
                "config": cfg.name,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "top_k": TOP_K,
                "question_id": qid,
                "expected": expected,
                "latency_sec": round(latency, 3),
                "index_build_sec": round(build_time, 3),
                "no_answer_pred": predicted_no,
                "no_answer_ok": correct_no_answer_behavior,
                "pages": ",".join(pages),
                "answer_preview": (answer[:180].replace("\n", " ") + ("..." if len(answer) > 180 else "")),
            })

    # CSV yaz
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Bitti. Rapor yazıldı: {OUTPUT_CSV}")
    print("İpucu: results.csv dosyasını Excel’de açıp latency/no_answer_ok karşılaştır.\n")


if __name__ == "__main__":
    main()
