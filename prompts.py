def create_prompt(query, results):
    context_parts = []
    for i, doc in enumerate(results['documents'][0], 1):
        doc_snippet = doc[:300] + "..." if len(doc) > 300 else doc
        context_parts.append(f"Excerpt {i}:\n{doc_snippet}\n")

    context = "\n".join(context_parts)
    
    prompt = f"""You are an AI assistant answering questions based on the provided excerpts.

Relevant Information:
{context}

Question: {query}

Instructions:
1. Answer based on the given excerpts if possible.
2. If the excerpts don't contain the specific information:
   a. Clearly state that the provided information doesn't answer the question directly.
   b. Then, provide a general or common answer based on your broad knowledge.
3. Cite excerpts when possible (e.g., "According to Excerpt 2...").
4. Be concise and to the point.
5. Always maintain a helpful and informative tone.

Answer:"""
    
    return prompt
