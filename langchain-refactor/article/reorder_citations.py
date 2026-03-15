import re
import os

def process_file(filepath, ext_mapping=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the bibliography section
    bib_start = content.find('\\begin{thebibliography}')
    bib_end = content.find('\\end{thebibliography}')
    if bib_start == -1 or bib_end == -1:
        print(f"Could not find bibliography in {filepath}")
        return content, ext_mapping

    body = content[:bib_start]
    bib_section = content[bib_start:bib_end]
    tail = content[bib_end:]

    # Parse bibitems
    # Format typically: \bibitem{b1} Author (Year)...
    bib_items = {}
    bib_pattern = re.compile(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\}|$)', re.DOTALL)
    for match in bib_pattern.finditer(bib_section):
        key = match.group(1).strip()
        text = match.group(2).strip()
        bib_items[key] = text

    # Find all citations in the body
    # Format: \cite{b1}, \cite{b1, b2, b3}
    cite_pattern = re.compile(r'\\cite\{([^}]+)\}')
    
    appearance_order = []
    seen = set()
    
    for match in cite_pattern.finditer(body):
        keys = [k.strip() for k in match.group(1).split(',')]
        for k in keys:
            if k not in seen:
                seen.add(k)
                appearance_order.append(k)
                
    # include any uncited items at the end
    for k in bib_items.keys():
        if k not in seen:
            appearance_order.append(k)
            seen.add(k)

    if ext_mapping is None:
        mapping = {old_k: f"b{i+1}" for i, old_k in enumerate(appearance_order)}
    else:
        mapping = ext_mapping

    # Replace citations in body
    def replace_cites(match):
        keys = [k.strip() for k in match.group(1).split(',')]
        new_keys = [mapping.get(k, k) for k in keys]
        # sort new keys based on the integer after 'b' if possible
        def get_num(key):
            try:
                return int(key[1:])
            except:
                return 999
        new_keys.sort(key=get_num)
        return '\\cite{' + ', '.join(new_keys) + '}'

    new_body = cite_pattern.sub(replace_cites, body)

    # Reconstruct bibliography
    new_bib_section = "\\begin{thebibliography}{00}\n"
    # sort the old keys by their new mapped integer
    def get_mapped_num(old_key):
        new_k = mapping.get(old_key, old_key)
        try:
            return int(new_k[1:])
        except:
            return 999
            
    sorted_old_keys = sorted(bib_items.keys(), key=get_mapped_num)
    
    for old_k in sorted_old_keys:
        new_k = mapping.get(old_k, old_k)
        new_bib_section += f"\\bibitem{{{new_k}}} {bib_items[old_k]}\n"

    new_content = new_body + new_bib_section + tail

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Processed {filepath}")
    return new_content, mapping

if __name__ == "__main__":
    base_dir = "/Users/brianrizqi/Documents/Post Doc/agentic-ai-native-ads/langchain-refactor/article"
    eng_file = os.path.join(base_dir, "agentic_native_ads.tex")
    ind_file = os.path.join(base_dir, "example.tex")
    
    _, mapping = process_file(eng_file)
    process_file(ind_file, mapping)
