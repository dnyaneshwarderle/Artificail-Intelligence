
import fitz
import pymupdf
from sentence_transformers import SentenceTransformer, util
from ollama import chat

# Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 1: Read PDF
def read_pdf(pdf_path):
    text = ""

    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"

    return text


# Step 2: Divide PDF text into chunks
def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Step 3: Accept PDF path
pdf_path = './AI-book.pdf'

pdf_text = read_pdf(pdf_path)

documents = create_chunks(pdf_text)

if not documents:
    raise ValueError("No extractable text found in PDF")

print("Total chunks:", len(documents))


# Step 4: Generate document embeddings
document_embedding = model.encode(
    documents,
    convert_to_tensor=True
)


# Step 5: Accept user question
query = input("Enter your question: ")


# Step 6: Generate query embedding
query_embedding = model.encode(
    query,
    convert_to_tensor=True
)


# Step 7: Calculate cosine similarity
scores = util.cos_sim(
    query_embedding,
    document_embedding
)[0]


# Step 8: Find top 2 relevant chunks
top_result = scores.topk(min(2, len(documents)))

retrieved_documents = []

print("\nTop relevant PDF chunks:")

for index in top_result.indices:
    chunk = documents[index.item()]

    retrieved_documents.append(chunk)

    print("\n", chunk)


# Step 9: Create context
context = "\n\n".join(retrieved_documents)


# Step 10: Create prompt
prompt = f"""
Answer the question using only the given PDF context.

Context:
{context}

Question:
{query}

If the answer is not available in context,
say "Information not available".

Answer:
"""


# Step 11: Send prompt to Ollama
response = chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0
    }
)

# Step 12: Display final answer
print("\nFinal Answer:")
print(response.message.content)
