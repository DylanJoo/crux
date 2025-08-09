import re
import string

def flatten_and_normalize(string_list):
    string = " ".join(string_list)
    return normalize_text(string)

def maybe_chunking(dlist, n=1024):
    overlength = [(i, len(d.split()) > n) for i, d in enumerate(dlist)]

    if any([o for _, o in overlength]):
        to_return = []
        for i, do_chunk in overlength:
            if do_chunk:
                words = dlist[i].split()
                while len(words) > 0:
                    to_return.append(" ".join(words[:512]))
                    words = words[512:]
            else:
                to_return.append(dlist[i])
        return to_return
    else:
        return dlist

def normalize_list(string_list):
    for i in range(len(string_list)):
        string_list[i] = normalize_text(string_list[i])
    return string_list

# def normalize(string):
def parse_mds(string):
    string = string.strip()
    string = normalize_text(string)
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')
