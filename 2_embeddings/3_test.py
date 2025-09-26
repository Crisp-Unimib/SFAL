import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# --- INITIAL SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

corpus = [
    "instances of the word \"false\" and its variations",
    "Occurrences of the word \"test\" in various contexts",
    "instances of the word \"text.\"",
    "occurrences of the word \"text\"",
    "method calls in programming that set or return values",
    "references to the movie \"Mockingjay\" and terms related to sockets",
    "elements related to testing and assertion in code",
]
query = "the term \"mock\" and its variations in various contexts"
print(f"Query: '{query}'\n")

# --- PHASE 1: RETRIEVAL ---
print("--- PHASE 1: RETRIEVAL (Candidate Retrieval) ---")
# Using a smaller, more manageable model as recommended
embedding_model_name = "Qwen/Qwen3-Embedding-4B"
embedding_model = SentenceTransformer(embedding_model_name, device=device)

corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True, device=device)
print(f"Corpus encoded into {corpus_embeddings.shape[0]} vectors.")
query_embedding = embedding_model.encode(query, prompt_name="query", convert_to_tensor=True, device=device)

k = 10
retrieval_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)[0]

print(f"\nResults of PHASE 1 (Retrieval) - Top {k}:")
candidate_docs = []
for hit in retrieval_hits:
    candidate_docs.append(corpus[hit['corpus_id']])
    print(f"- Score: {hit['score']:.4f} | Document: '{corpus[hit['corpus_id']]}'")

# --- AGGRESSIVE MEMORY CLEANUP ---
if device == "cuda":
    print("\nAggressively cleaning GPU memory...")
    del query_embedding
    del corpus_embeddings
    embedding_model = embedding_model.cpu()
    del embedding_model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleaned.\n")

# --- PHASE 2: RERANKING ---
print("--- PHASE 2: RERANKING (Precision Reordering) ---")
# Using a smaller, more manageable model
reranker_model_name = "Qwen/Qwen3-Reranker-4B"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model_name).to(device).eval()

token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = """<|im_start|>system
Your task is to act as a semantics expert. Judge whether the Document is **conceptually and semantically related** to the Query, even if the words used are different. **Ignore superficial or structural similarity** and focus on the deep meaning.
Here are some examples of how to judge:
---
**Example 1 (Good semantic match, different structure):**
Query: the word false and its derivations
Document: the concept of falsehood and being untrue
Answer: yes
---
**Example 2 (Bad semantic match, identical structure):**
Query: the word false and its derivations
Document: the word house and its derivations
Answer: no
---
**Example 3 (Good semantic match, no shared keywords):**
Query: companies that build skyscrapers
Document: firms specializing in the construction of very tall buildings
Answer: yes
---
**Example 4 (Bad semantic match, shared keyword but different context):**
Query: the history of the Apple computer
Document: the nutritional value of an apple
Answer: no
---
Now, based on this understanding, judge the following pair. Your answer must be only 'yes' or 'no'.<|im_end|>
<|im_start|>user
"""
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

# ==================== MODIFIED FUNCTION ====================
def format_instruction(query, doc, instruction=None):
    """
    Formats the user prompt. If an instruction is provided, it includes it.
    Otherwise, it omits the instruction block for a cleaner prompt.
    """
    if instruction and instruction.strip():
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    else:
        return f"<Query>: {query}\n<Document>: {doc}"
# =========================================================

@torch.no_grad()
def compute_rerank_score(query, doc, instruction=None):
    # This now uses the new, flexible format_instruction function
    pair_text = format_instruction(query, doc, instruction=instruction)
    
    inputs = reranker_tokenizer(
        [pair_text], padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
        
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    
    stacked_scores = torch.stack([false_vector, true_vector], dim=1)
    log_softmax_scores = torch.nn.functional.log_softmax(stacked_scores, dim=1)
    
    score = log_softmax_scores[:, 1].exp().item()
    return score

rerank_pairs = [(query, doc) for doc in candidate_docs]
print("Calculating Reranking scores...")
rerank_scores = []
# The `task_instruction` is no longer needed, as the system prompt handles everything.
# We will call compute_rerank_score without the instruction argument.
for q, doc in rerank_pairs:
    score = compute_rerank_score(q, doc, instruction=None) # Passing None to use the cleaner prompt
    rerank_scores.append({'document': doc, 'score': score})

rerank_scores.sort(key=lambda x: x['score'], reverse=True)
print("\nFINAL Results after PHASE 2 (Reranking):")
for item in rerank_scores:
    print(f"- Score: {item['score']:.4f} | Document: '{item['document']}'")