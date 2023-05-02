import torch
import os
import re
from clip.simple_tokenizer import SimpleTokenizer

from .prompt_template import prompt_templates

class SentPreProcessor(object):
    def __init__(self, root, loaded_idxs, label_mapping_file, wiki_dir, context_length=75):
        self.root = root
        self.loaded_idxs = loaded_idxs
        self.label_mapping_file = label_mapping_file
        self.wiki_dir = wiki_dir

        self.drop_keys = ['External links', 'References', 'Further reading', 'Bibliography']
        self._tokenizer = SimpleTokenizer()
        self.SEP_TOKENS = [267, 269] # [',', '.']
        self.context_length = context_length

    def _get_wiki_hash(self):
        with open(os.path.join(self.root, self.label_mapping_file), "r") as rf:
            data = rf.readlines()
        hash_table = {}
        _lines = [l.split() for l in data]
        for l in _lines:
            if l[0].strip() in self.loaded_idxs:
                hash_table[l[0].strip()] = [l[1].strip(), l[-1].strip().replace("_", " ")]
        assert len(hash_table.keys()) == len(self.loaded_idxs), "There are no matching indices in label mapping file. Please use valid label mapping file."
        return hash_table

    def _parse_desc(self, desc_path):
        try:
            with open(desc_path) as rf:
                lines = rf.readlines()
        except UnicodeDecodeError:
            with open(desc_path, encoding='gbk') as rf:
                lines = rf.readlines()
        lines = [d.strip() for d in lines if d.strip() != '']
        ret_dict = {}
        key = "summary"
        val = ""
        for line in lines:
            if line[:2] == "==":
                ret_dict[key] = val.strip()
                key = line.strip('= ')
                val = ""
            else:
                val += line + '\n'
        ret_dict[key] = val.strip()
        return ret_dict
    
    def _gen_naive_desc(self, name):
        texts = [template.format(name + ' ') for template in prompt_templates]
        return '\n'.join(texts)
    
    def _get_text(self, wiki):
        # use all key part of each wiki text except those in drop_keys
        text = wiki["summary"] + "\n"
        text += "\n".join([v for k, v in wiki.items() if k not in ["summary"] + self.drop_keys])
        return text

    def _split_sent(self, text):
        pat = re.compile(r'(?<!\w\.\w.)(?<!([A-Z][a-z])|([A-Z])\.)(?<=\.|\?)(?=[\sA-Z])', re.X)

        split_text = pat.split(text)
        split_text = [s.strip() for s in split_text if s is not None and s.strip() != '']
        return split_text

    def tokenize(self, texts):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]  # 49406
        eot_token = self._tokenizer.encoder["<|endoftext|>"]  # 49407
        all_tokens = [[sot_token] + self._tokenizer.encode(text)[:self.context_length] + [eot_token] for text in
                        texts]
        result = torch.zeros(len(all_tokens), self.context_length + 2, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length + 2:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {self.context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def make_sentence_tokens(self):
        hash_table = self._get_wiki_hash()

        sentence_tokens = []
        wiki_dir = os.path.join(self.root, self.wiki_dir)
        for idxs in self.loaded_idxs:
            now_wiki_desc = os.path.join(wiki_dir, f"desc_{hash_table[idxs][0]}.txt")
            wiki_desc = self._parse_desc(now_wiki_desc)
            wiki_text = self._get_text(wiki_desc)
            naive_text = self._gen_naive_desc(hash_table[idxs][1])
            text = naive_text + wiki_text

            splited_texts = self._split_sent(text)
            sentence_tokens.append(self.tokenize(splited_texts))

        return sentence_tokens