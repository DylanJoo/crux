# system_prompt = "You are a writing assistant that can write an accurate, faithful, and concise answer for the given question.\n\nInstruction: Use only the provided documents (some of which might be irrelevant) as references. Cite every sentences with at least one document in the citation format of [number], or [number1][number2] when citing multiple documents. One sentence has at most three documents cited. The citation is good to be at the end of the sentence. No need to list the citation after the entire report."

# instruction_prompt_alce = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."

instruction_prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided documents as references (some of which might be irrelevant). Always cite at least one document for every sentences in the answer. Use the citation format of square brackets to indicate the cited documents (e.g., [1] for the first reference). If multiple documents support the sentence, only cite a minimum sufficient subset of the documents, with the citation format like [1][2][3]. Only generate the answer, excluding any disclaimers, notes or list of references at the end of the answer."

demo_sep = "\n\n" # the results so far used three seps.
doc_prompt_template = "Document [{ID}]: (Title: {T}) {P}\n"
demo_prompt_template = "{INST}\n\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
inst_prompt_template = "{INST}\n\n\n{DEMO}Question: {Q}\n\n{D}\nAnswer: {A}"

# Instruction: {INST} \n\n\n
# Question: {Q} \n
# {Document [0]: {D0}\n
# Document [1]: {D1}\n
# ... 
# Document [k]: {Dk}\n\n
# Answer: {A} \n\n
# ( if using demo, it would be the one above.)
# Question: {Q}           
# {Document [0]: {D0}
# Document [1]: {D1}
# ... 
# Document [k]: {Dk}
# Answer: {A}

def apply_docs_prompt(doc_items, ndoc=None, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items[:ndoc]):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item['title'])
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def apply_demo_prompt(Q, D, A, instruction=""):
    p = demo_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", A)
    return p

def apply_inst_prompt(Q, D, instruction="", add_prefix=True):
    p = inst_prompt_template
    p = p.replace("{INST}", instruction).strip()
    p = p.replace("{Q}", Q).replace("{D}", D).replace("{A}", "")
    if add_prefix is False:
        p = p.replace("Answer:", "").strip()
    return p
