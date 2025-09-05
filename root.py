import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. Read full text ===
with open("insight_text.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# === 2. Split entries using "Heading:" as separator ===
entries = full_text.split("Heading:")
entries = [entry.strip() for entry in entries if entry.strip()]

# === 3. Prepare splitter ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

all_documents = []

for entry in entries:
    # === 4. Extract heading line ===
    lines = entry.splitlines()
    heading_line = lines[0].strip()
    body_text = "\n".join(lines[1:]).strip()

    # === 5. Parse name and company from heading ===
    # Example: "Abhay Gupta: Placed at OLA"
    if ":" in heading_line and "Placed at" in heading_line:
        name_part, placed_part = heading_line.split(":", 1)
        name = name_part.strip()
        company = placed_part.replace("Placed at", "").strip()
    else:
        # fallback if heading malformed
        name = "Unknown"
        company = "Unknown"

    metadata = {
        "name": name,
        "company": company
    }

    # === 6. Chunk and add metadata ===
    chunks = text_splitter.split_text(body_text)
    docs = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
    all_documents.extend(docs)

# === 7. Build and save FAISS ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_documents, embeddings)
vectorstore.save_local("faiss_index_insights2")

print("✅ FAISS vectorstore created with", len(all_documents), "chunks.")
