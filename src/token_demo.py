import tiktoken

# enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 用的编码
enc = tiktoken.get_encoding("o200k_base")  # GPt-4o编码，对中文效率更高
text = "今天天气真好，我想出去玩。"
tokens = enc.encode(text)
print(f"文本: {text}")
print(f"token 数: {len(tokens)}")
print(f"每个 token 对应: {[enc.decode([t]) for t in tokens]}")
print(tokens)
for i in tokens:
    print(i, repr(enc.decode([i])))
for i in tokens:
    print(i, enc.decode_single_token_bytes(i))
print(enc.decode(tokens))
