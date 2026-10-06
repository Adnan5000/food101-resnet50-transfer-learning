# Food-101 · ResNet-50 transfer learning

Fine-tuning ResNet-50 on [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) —
101 food classes, 101,000 photos — to **83.46% top-1 / 96.35% top-5** on the
official held-out test set, in 54 minutes on a laptop GPU.

Comes with a web page you can drop a photo into, so you can see what it actually
does rather than take the numbers on faith.

> The same recipe on ResNet-18 is in
> [food101-resnet18-transfer-learning](https://github.com/Adnan5000/food101-resnet18-transfer-learning):
> 75.98% top-1, trains in 16 minutes, 43 MB on disk. This repo is the accuracy
> end of that trade.

## Results

Measured on Food-101's **official** 25,250-image test split — the standard one,
so these are comparable to published numbers. The model never trained on any of it.

| Metric | Value |
|---|---|
| Top-1 accuracy | **83.46%** |
| Top-5 accuracy | **96.35%** |
| Trainable parameters | 23,714,981 |
| Training time | 53.9 min (5 epochs, RTX 3060 Laptop, 6GB) |
| Throughput | ~131 img/s at batch 32 |
| Checkpoint size | 90.8 MB |

Per epoch:

| Epoch | train loss | train acc | **val top-1** | val top-5 |
|---:|---:|---:|---:|---:|
| 1 | 3.210 | 38.3% | 70.10% | 90.96% |
| 2 | 1.915 | 68.0% | 78.63% | 94.67% |
| 3 | 1.690 | 74.9% | 81.79% | 95.82% |
| 4 | 1.579 | 78.3% | 83.31% | 96.30% |
| 5 | 1.506 | 80.7% | **83.46%** | **96.35%** |

The last epoch added only **+0.15**, so this is close to the ceiling for this
recipe. More epochs would buy a point or two, not ten.
Raw numbers in [`training_history.json`](training_history.json).

**Top-1 vs top-5:** top-1 is "the single best guess was right." Top-5 is "the
right answer was in the first five." Top-5 is always higher and it is not the
headline number, though it is often quoted as one.

### Against ResNet-18, same recipe

Identical data, schedule, augmentation and epoch count — only the backbone and
the batch size differ, so this is a clean comparison.

| | ResNet-18 | ResNet-50 |
|---|---|---|
| Top-1 | 75.98% | **83.46%** (+7.5) |
| Top-5 | 93.49% | **96.35%** (+2.9) |
| Parameters | 11.2M | 23.7M (2.1×) |
| Training time | 15.6 min | 53.9 min (3.4×) |
| Checkpoint | 42.9 MB | 90.8 MB |

ResNet-50 passed ResNet-18's *final* score at **epoch 2**. Whether 7.5 points is
worth 3.4× the training time and 2.1× the disk depends entirely on where it runs —
on a phone, ResNet-18 is the better answer.

## Try it on your own photos

You need Python 3.12+ and about 1.5 GB of disk for PyTorch. No GPU required —
inference runs fine on CPU at ~60 ms per image.

```bash
git clone https://github.com/Adnan5000/food101-resnet50-transfer-learning
cd food101-resnet50-transfer-learning

python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Download `resnet50_food101.pt` from
[**Releases**](https://github.com/Adnan5000/food101-resnet50-transfer-learning/releases)
into a `weights/` folder (it is 90.8 MB, kept out of the repo so clones stay fast):

```bash
mkdir -p weights
# put resnet50_food101.pt in weights/
uvicorn serve:app --port 8001
```

Open <http://localhost:8001>, drag in a photo of food, and you get the top five
guesses with confidences and the inference time.

There is also a plain HTTP endpoint:

```bash
curl -X POST "http://localhost:8001/v1/classify-upload?top_k=5" -F "file=@dinner.jpg"
```

```json
{
  "predictions": [
    { "class": "pancakes",         "confidence": 0.9056 },
    { "class": "huevos_rancheros", "confidence": 0.0699 },
    { "class": "cheesecake",       "confidence": 0.0035 }
  ],
  "inference_ms": 58
}
```

## Training it yourself

```bash
pip install -r requirements-train.txt
python train.py --cache-dir ./data --epochs 5
```

First run downloads Food-101 (~5 GB) and builds a ~10 GB cache.

Batch size defaults to **32**, not 64: ResNet-50 holds much larger activations
and 64 does not fit 6 GB of VRAM at 224px. Raise it if you have more.

## What actually moved the number

Starting point was a frozen backbone — ImageNet weights held fixed, training only
the 101-way classifier on top. That plateaus at **~44%** no matter how long you
run it. Two changes account for almost all the gain:

**1. Unfreezing the backbone.** A frozen network is a linear probe over features
built for ImageNet's 1000 classes. Food-101 is fine-grained — the model has to
separate red velvet cake from carrot cake — and that needs the convolutional
filters themselves to adapt. Training the full network means 23,714,981
parameters instead of 206,949.

**2. Using the whole dataset.** The frozen run trained on 100 images per class.
Food-101 ships 750 per class, so 89% was being thrown away. 8,080 → 75,750 images.

Smaller contributions: `IMAGENET1K_V2` pretrained weights rather than V1; SGD with
Nesterov momentum and a OneCycle schedule instead of constant-LR Adam; the
backbone at a 10× lower learning rate than the randomly-initialised head, since
pretrained features need gentler updates; label smoothing 0.1; weight decay 1e-4;
augmentation with a 0.5 crop-scale floor and colour jitter; mixed precision.

Note the final epoch shows train accuracy (80.7%) *below* validation (83.46%).
That is not a bug — training images get deliberately distorted while validation
images are shown clean, and the training figure is a running average over an
epoch that started much worse.

## Two things that would have been silent bugs

**Class ordering.** The HuggingFace mirror of Food-101 orders its labels
differently from the canonical `classes.txt` — `cheesecake` before
`cheese_plate`. Trusting the index would have transposed those two classes with
no error anywhere. `train.py` asserts the sets match and remaps the indices, and
the checkpoint stores its own label list so inference cannot pair weights with
the wrong order.

**The download host.** `torchvision.datasets.Food101` points at
data.vision.ee.ethz.ch, which measured **0.17 MB/s** during development — about
seven hours for 4.65 GB. The HuggingFace mirror served the same data at 3.4 MB/s.
Same images, same official split.

## Honest limits

- **Food-101 only.** 101 classes. Anything else gets confidently mapped to the
  nearest of those 101 — there is no "not food" output.
- **Confident errors are the real failure mode.** Red velvet cake comes back as
  carrot cake at 0.70. A near-tie is recoverable by showing the user alternatives;
  a confident mistake gives the interface no signal at all. Accuracy does not
  capture this, and it matters more than the headline number for anything that
  acts on the output. A per-class confusion matrix is the right next diagnostic.
- Interestingly, ResNet-50 is *less* confident on several answers it gets right
  (macaroni cheese 0.90 → 0.52 versus ResNet-18). That is label smoothing plus
  better calibration, not a regression.
- **Single centre crop at inference.** Test-time augmentation would add a point
  or so at several times the cost.
- Trained on one 6 GB laptop GPU for 54 minutes. Not a leaderboard attempt —
  EfficientNet and ConvNeXt backbones do better on this benchmark.

## Licence

Code is MIT. The Food-101 dataset and weights derived from it are ETH Zurich's
and published for research use — see [LICENSE](LICENSE).
