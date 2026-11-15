"""Frozen encoder; anonymous tokens are never passed to its tokenizer."""
import gc
import hashlib
import json
import numpy as np
import torch
from regime_inspector import ROOT, PREVIOUS, vocabulary, verbalize

def attach_relations(z, names):
    from dataset_inspector import relation_texts
    from regime_inspector import readable
    cleaned = relation_texts([x if readable(x) else '' for x in names])
    for i, name in enumerate(names):
        if not readable(name):
            cleaned[i] = cleaned[i + len(names)] = None
    result = encode_names(cleaned, 'relations')
    if result is not None:
        z['_lex_rel'] = result

def embeddings(rec, z):
    names, relations, _ = vocabulary(rec, z)
    result = encode_names(names, rec['key'])
    if result is not None:
        attach_relations(z, relations)
    return result

def encode_names(names, key='independent-validation'):
    texts = [verbalize(x) if x is not None else None for x in names]
    eligible = [i for i, t in enumerate(texts) if t is not None]
    if not eligible:
        return None
    identity = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    directory = ROOT / 'cache/text'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (identity + '.npy')
    if path.exists():
        return np.load(path)
    # Existing ConceptNet cache is aligned to the public vocabulary by its identity.
    if key == 'ConceptNet100k_external':
        from dataset_inspector import load_vocab, relation_texts
        _, original, relations = load_vocab()
        old = torch.load(PREVIOUS / 'cache/conceptnet_text_emb.pt', map_location='cpu', weights_only=False)
        expected = hashlib.sha256(json.dumps(original + relation_texts(relations)).encode()).hexdigest()
        assert old['identity'] == expected
        output = old['entities'].numpy().copy()
        # Never use pre-existing encodings for numeric/opaque names.
        for i, text in enumerate(original):
            if verbalize(text) is None:
                output[i] = 0
        np.save(path, output)
        return output
    from semantic_calibrator import ENCODER
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ENCODER), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(ENCODER), local_files_only=True).cuda().half().eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    output = np.zeros((len(names), 384), np.float32)
    with torch.inference_mode():
        for at in range(0, len(eligible), 128):
            ids = eligible[at:at + 128]
            batch = tokenizer([texts[i] for i in ids], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
            encoded = torch.nn.functional.normalize(encoder(**batch).last_hidden_state[:, 0].float(), dim=-1)
            output[ids] = encoded.cpu().numpy()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    np.save(path, output)
    return output
