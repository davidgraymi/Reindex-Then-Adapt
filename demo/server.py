"""Live backend for the RTA demo: runs Llama-2-7b-chat to encode the
conversation exactly like tools/3_context_encoding.py did for the datasets,
then scores items with the trained aggregator + adapt biases.

Serves demo/ statics plus:
    GET  /api/status     -> {"ready": bool, "device": str, ...}
    POST /api/recommend  -> {"turns": [{"speaker","text"},...], "k": 10}
                            => {"before": [...], "after": [...]}

Examples:
    .venv/bin/python demo/server.py                  # http://localhost:8765
    .venv/bin/python demo/server.py --device cpu --self_test true
"""

import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "tools"))

# allow the fp16 model to exceed the default MPS memory watermark on 16GB
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from jsonargparse import CLI

from demo import (
    build_item_embeddings,
    find_adapt_ckpt,
    load_aggregator,
    load_biases,
)

LLM_NAME = "NousResearch/Llama-2-7b-chat-hf"  # ungated mirror of meta-llama
PROMPT_PREFIX = (
    "<s>[INST] <<SYS>>\nPretend you are a movie recommender system. \n"
    "I will give you a conversation between a user and you (a recommender"
    " system). Based on the conversation, you reply me with 20 movie titles"
    " without extra sentences.\n\n<</SYS>>\n\nHere is the conversation: "
)
PROMPT_SUFFIX = (
    "\n [/INST]  Sure, here are 20 movie titles that I would recommend"
    " based on the conversation:\n1."
)

STATE = {"ready": False, "status": "loading recommender...", "device": None}
LOCK = threading.Lock()


class Pipeline:
    def __init__(self, dataset, device):
        self.k = 10
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

        STATE["status"] = "loading aggregator + adapt biases..."
        aggregator = load_aggregator(
            os.path.join(ROOT_DIR, "ckpts/best_aggregator/aggregator.pt")
        )
        add_bias, mul_bias = load_biases(find_adapt_ckpt(dataset))

        mapping = json.load(
            open(os.path.join(ROOT_DIR, "data/item_entity_name_to_item_indices.json"))
        )
        names = json.load(
            open(os.path.join(ROOT_DIR, f"data/{dataset}/id2name.json"))
        ).values()
        self.candidates = torch.LongTensor(
            sorted({mapping[n]["single_token"] for n in names})
        )
        self.token2name = {
            m["single_token"]: m["friendly_title"] for m in mapping.values()
        }
        single2llm = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(m["llm_tokens"]) for m in mapping.values()],
            batch_first=True,
            padding_value=0,
        )
        order = torch.LongTensor([m["single_token"] for m in mapping.values()])
        single2llm = single2llm[order.argsort()]

        STATE["status"] = "building item embeddings..."
        llm_embed = torch.load(
            os.path.join(ROOT_DIR, "data/llm_embedding.pt"), map_location="cpu"
        )
        with torch.no_grad():
            self.item_embed = build_item_embeddings(
                aggregator, self.candidates, single2llm, llm_embed
            )
        self.cand_add = add_bias[self.candidates]
        self.cand_mul = mul_bias[self.candidates]
        del llm_embed, aggregator

        STATE["status"] = f"loading {LLM_NAME} on {self.device} (first time is slow)..."
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(LLM_NAME)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        dtype = torch.float16 if self.device == "mps" else torch.bfloat16
        self.llm = (
            transformers.AutoModelForCausalLM.from_pretrained(
                LLM_NAME, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            .to(self.device)
            .eval()
        )
        STATE["ready"] = True
        STATE["device"] = self.device
        STATE["status"] = "ready"

    @torch.no_grad()
    def encode(self, prompt):
        """Last hidden state at the last prompt token (matches dataset gen)."""
        inputs = self.tokenizer(
            [prompt], truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        inputs["position_ids"] = (
            inputs["attention_mask"].cumsum(dim=-1) - 1
        ).relu()
        hidden = self.llm.model(**inputs).last_hidden_state
        return hidden[0, -1, :].float().cpu()

    @torch.no_grad()
    def recommend(self, turns, k=10):
        convo = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        q = self.encode(PROMPT_PREFIX + convo + PROMPT_SUFFIX)
        base = q @ self.item_embed.T
        adapted = (base + self.cand_add) * self.cand_mul

        def top(scores):
            order = scores.argsort(descending=True)[:k]
            return [
                {
                    "name": self.token2name.get(
                        self.candidates[c].item(), "<unknown>"
                    ),
                    "token": self.candidates[c].item(),
                    "score": round(scores[c].item(), 3),
                    "gt": False,
                }
                for c in order.tolist()
            ]

        return {"before": top(base), "after": top(adapted)}

    def self_test(self, dataset):
        """Re-encode a stored test turn and compare to its dataset encoding."""
        with open(
            os.path.join(ROOT_DIR, f"data/{dataset}/test-for-adapt.jsonl")
        ) as f:
            sample = json.loads(f.readline())
        mine = self.encode(sample["encoded_text"])
        ref = torch.tensor(sample["encoding"])
        cos = torch.nn.functional.cosine_similarity(mine, ref, dim=0).item()
        print(f"self-test cosine(live encoding, dataset encoding) = {cos:.4f}")
        return cos


PIPELINE = None


def make_handler(dataset):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=CUR_DIR, **kw)

        def log_message(self, fmt, *args):
            sys.stderr.write("%s\n" % (fmt % args))

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                return self._json(200, STATE)
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/recommend":
                return self._json(404, {"error": "unknown endpoint"})
            if not STATE["ready"]:
                return self._json(503, {"error": STATE["status"]})
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or "{}")
            turns = req.get("turns", [])
            if not turns:
                return self._json(400, {"error": "no turns"})
            with LOCK:
                result = PIPELINE.recommend(turns, int(req.get("k", 10)))
            return self._json(200, result)

    return Handler


def main(
    dataset: str = "inspired",
    port: int = 8765,
    device: str = "auto",
    self_test: bool = False,
):
    global PIPELINE

    def load():
        global PIPELINE
        try:
            PIPELINE = Pipeline(dataset, device)
            if self_test:
                PIPELINE.self_test(dataset)
        except Exception as e:
            STATE["status"] = f"load failed: {e}"
            raise

    threading.Thread(target=load, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(dataset))
    print(f"RTA demo server: http://localhost:{port} (dataset={dataset})")
    server.serve_forever()


if __name__ == "__main__":
    CLI(main)
