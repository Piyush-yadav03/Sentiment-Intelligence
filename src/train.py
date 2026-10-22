import argparse, json, random, os, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinyTransformerStudent
OUT = ROOT / "results"
TEACHER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SEED = 42

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

class Reviews(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, padding="max_length",
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i], "labels": self.labels[i]}

class DistilledReviews(Dataset):
    def __init__(self, base, logits): self.base, self.logits = base, logits
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        item = self.base[i]; item["teacher_logits"] = self.logits[i]; return item

def balanced(split, size):
    half = size // 2
    neg = [t for t, y in zip(split["text"], split["label"]) if y == 0][:half]
    pos = [t for t, y in zip(split["text"], split["label"]) if y == 1][:half]
    texts, labels = neg + pos, [0]*len(neg) + [1]*len(pos)
    idx = list(range(len(labels))); random.Random(SEED + size).shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]

@torch.no_grad()
def teacher_logits(model, loader):
    model.eval(); out = []
    for b in tqdm(loader, desc="Teacher logits"):
        out.append(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.cpu())
    return torch.cat(out)

def train(model, loader, epochs, mode, temperature, alpha):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    for ep in range(epochs):
        model.train(); total = 0
        for b in tqdm(loader, desc=f"{mode} {ep+1}/{epochs}"):
            logits = model(b["input_ids"], b["attention_mask"])
            hard = F.cross_entropy(logits, b["labels"])
            if mode == "distilled":
                soft = F.kl_div(F.log_softmax(logits/temperature, -1),
                    F.softmax(b["teacher_logits"]/temperature, -1),
                    reduction="batchmean") * temperature**2
                loss = alpha*hard + (1-alpha)*soft
            else: loss = hard
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(b["labels"])
        print(f"{mode} epoch {ep+1}: loss {total/len(loader.dataset):.4f}")
    return model

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); y, p = [], []
    for b in loader:
        p.extend(model(b["input_ids"], b["attention_mask"]).argmax(-1).tolist())
        y.extend(b["labels"].tolist())
    return {"accuracy": float(accuracy_score(y,p)),
            "macro_f1": float(f1_score(y,p,average="macro"))}

def latency(model, loader):
    model.eval(); times=[]; count=0
    with torch.no_grad():
        for i,b in enumerate(loader):
            if i < 2: model(b["input_ids"], b["attention_mask"]); continue
            start=time.perf_counter(); model(b["input_ids"], b["attention_mask"])
            times.append(time.perf_counter()-start); count += len(b["labels"])
            if len(times) >= 10: break
    return 1000*sum(times)/max(count,1)

def new_student(tokenizer, max_len):
    return TinyTransformerStudent(len(tokenizer), tokenizer.pad_token_id or 0, max_length=max_len)

def save(model, path):
    torch.save({"state_dict": model.cpu().state_dict()}, path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-size",type=int,default=800)
    ap.add_argument("--test-size",type=int,default=300)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--max-length",type=int,default=96)
    ap.add_argument("--temperature",type=float,default=2.0)
    ap.add_argument("--alpha",type=float,default=.5)
    a=ap.parse_args()
    if a.train_size < 2 or a.test_size < 2 or a.epochs < 1: raise ValueError("Sizes must be >=2 and epochs >=1")
    seed_all(); OUT.mkdir(exist_ok=True)
    print("Loading IMDb (first run downloads dataset)...")
    ds=load_dataset("imdb")
    tr_x,tr_y=balanced(ds["train"],a.train_size); te_x,te_y=balanced(ds["test"],a.test_size)
    tok=AutoTokenizer.from_pretrained(TEACHER)
    teacher=AutoModelForSequenceClassification.from_pretrained(TEACHER).eval()
    tok.save_pretrained(OUT/"tokenizer")
    tr=Reviews(tr_x,tr_y,tok,a.max_length); te=Reviews(te_x,te_y,tok,a.max_length)
    tr_eval=DataLoader(tr,batch_size=a.batch_size,shuffle=False)
    tr_loader=DataLoader(tr,batch_size=a.batch_size,shuffle=True)
    te_loader=DataLoader(te,batch_size=a.batch_size,shuffle=False)
    start=time.perf_counter(); logits=teacher_logits(teacher,tr_eval); teacher_secs=time.perf_counter()-start
    torch.save(logits,OUT/"teacher_logits.pt")
    dist_loader=DataLoader(DistilledReviews(tr,logits),batch_size=a.batch_size,shuffle=True)

    print("Train baseline student")
    base=train(new_student(tok,a.max_length),tr_loader,a.epochs,"baseline",a.temperature,a.alpha)
    save(base,OUT/"student_baseline.pt")
    print("Train distilled student")
    student=train(new_student(tok,a.max_length),dist_loader,a.epochs,"distilled",a.temperature,a.alpha)
    save(student,OUT/"student_distilled.pt")
    b=evaluate(base,te_loader); d=evaluate(student,te_loader)
    b["latency_ms_per_sample"]=latency(base,te_loader)
    d["latency_ms_per_sample"]=latency(student,te_loader)
    b["checkpoint_mb"]=(OUT/"student_baseline.pt").stat().st_size/1024**2
    d["checkpoint_mb"]=(OUT/"student_distilled.pt").stat().st_size/1024**2
    results={"teacher_model":TEACHER,"dataset":"IMDb","device":"CPU","seed":SEED,
      "train_examples":len(tr),"test_examples":len(te),"epochs":a.epochs,"batch_size":a.batch_size,
      "max_length":a.max_length,"temperature":a.temperature,"alpha_hard_loss":a.alpha,
      "teacher_logits_generation_seconds":teacher_secs,"baseline_student":b,"distilled_student":d,
      "limitations":["Teacher is fine-tuned on SST-2 but evaluated on IMDb (domain shift).",
                     "Subset results are not full-dataset benchmark results.",
                     "CPU latency depends on hardware and system load."]}
    (OUT/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({"baseline":b,"distilled":d},indent=2))
    print("Done. Results:",OUT.resolve())

if __name__=="__main__": main()
