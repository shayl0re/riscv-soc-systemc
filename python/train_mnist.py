"""Train the paper's 8-layer MNIST CNN with plain NumPy (no PyTorch needed).
conv3x3(32) -> ReLU -> maxpool2 -> conv3x3(64) -> ReLU -> maxpool2 -> FC 1600->128 -> ReLU -> FC 128->10
Output: model_float.npz
"""
import gzip, os, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

D = os.path.join(os.path.dirname(__file__), "..", "data")
rng = np.random.default_rng(0)


def load_images(f):
    with gzip.open(os.path.join(D, f)) as g:
        return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def load_labels(f):
    with gzip.open(os.path.join(D, f)) as g:
        return np.frombuffer(g.read(), np.uint8, offset=8)


def conv_fwd(x, W, b):
    N, C, H, Wd = x.shape
    F = W.shape[0]
    Ho, Wo = H - 2, Wd - 2
    win = swv(x, (3, 3), axis=(2, 3))                       # N,C,Ho,Wo,3,3
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * 9)
    out = cols @ W.reshape(F, -1).T + b
    return out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2), cols


def conv_bwd(dout, cols, W, xshape):
    N, C, H, Wd = xshape
    F = W.shape[0]
    Ho, Wo = H - 2, Wd - 2
    d2 = dout.transpose(0, 2, 3, 1).reshape(-1, F)
    dW = (d2.T @ cols).reshape(W.shape)
    db = d2.sum(0)
    dc = (d2 @ W.reshape(F, -1)).reshape(N, Ho, Wo, C, 3, 3)
    dx = np.zeros(xshape, np.float32)
    for i in range(3):
        for j in range(3):
            dx[:, :, i:i + Ho, j:j + Wo] += dc[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return dx, dW, db


def pool_fwd(x):
    N, C, H, W = x.shape
    Ho, Wo = H // 2, W // 2
    xc = x[:, :, :Ho * 2, :Wo * 2].reshape(N, C, Ho, 2, Wo, 2)
    out = xc.max(axis=(3, 5))
    return out


def pool_bwd(dout, x):
    N, C, H, W = x.shape
    Ho, Wo = H // 2, W // 2
    xc = x[:, :, :Ho * 2, :Wo * 2].reshape(N, C, Ho, 2, Wo, 2)
    out = xc.max(axis=(3, 5), keepdims=True)
    mask = (xc == out)
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)
    dxc = mask * dout[:, :, :, None, :, None]
    dx = np.zeros_like(x)
    dx[:, :, :Ho * 2, :Wo * 2] = dxc.reshape(N, C, Ho * 2, Wo * 2)
    return dx


def init(shape, fan_in):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


P = {
    "w1": init((32, 1, 3, 3), 9), "b1": np.zeros(32, np.float32),
    "w2": init((64, 32, 3, 3), 288), "b2": np.zeros(64, np.float32),
    "w3": init((128, 1600), 1600), "b3": np.zeros(128, np.float32),
    "w4": init((10, 128), 128), "b4": np.zeros(10, np.float32),
}


def forward(x, keep=False):
    z1, c1 = conv_fwd(x, P["w1"], P["b1"]); a1 = np.maximum(z1, 0); p1 = pool_fwd(a1)
    z2, c2 = conv_fwd(p1, P["w2"], P["b2"]); a2 = np.maximum(z2, 0); p2 = pool_fwd(a2)
    f = p2.reshape(len(x), -1)
    z3 = f @ P["w3"].T + P["b3"]; a3 = np.maximum(z3, 0)
    z4 = a3 @ P["w4"].T + P["b4"]
    if keep:
        return z4, (x, c1, a1, p1, c2, a2, p2, f, a3)
    return z4


def main(epochs=3, bs=128, lr=2e-3):
    Xtr = load_images("train-images-idx3-ubyte.gz").astype(np.float32)[:, None] / 255
    ytr = load_labels("train-labels-idx1-ubyte.gz")
    Xte = load_images("t10k-images-idx3-ubyte.gz").astype(np.float32)[:, None] / 255
    yte = load_labels("t10k-labels-idx1-ubyte.gz")
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(x) for k, x in P.items()}
    t = 0
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr)); t0 = time.time()
        for s in range(0, len(idx), bs):
            bi = idx[s:s + bs]; x = Xtr[bi]; y = ytr[bi]; n = len(bi)
            z4, (x, c1, a1, p1, c2, a2, p2, f, a3) = forward(x, True)
            e = np.exp(z4 - z4.max(1, keepdims=True)); sm = e / e.sum(1, keepdims=True)
            d4 = sm.copy(); d4[np.arange(n), y] -= 1; d4 /= n
            g = {}
            g["w4"] = d4.T @ a3; g["b4"] = d4.sum(0)
            d3 = (d4 @ P["w4"]) * (a3 > 0)
            g["w3"] = d3.T @ f; g["b3"] = d3.sum(0)
            df = (d3 @ P["w3"]).reshape(p2.shape)
            da2 = pool_bwd(df, a2) * (a2 > 0)
            dp1, g["w2"], g["b2"] = conv_bwd(da2, c2, P["w2"], p1.shape)
            da1 = pool_bwd(dp1, a1) * (a1 > 0)
            _, g["w1"], g["b1"] = conv_bwd(da1, c1, P["w1"], x.shape)
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if (s // bs) % 100 == 0:
                print(f"ep{ep} step{s // bs} loss {-np.log(sm[np.arange(n), y] + 1e-9).mean():.3f} {time.time() - t0:.0f}s", flush=True)
        lr *= 0.5
        acc = np.mean(np.concatenate([forward(Xte[i:i + 500]).argmax(1) for i in range(0, 10000, 500)]) == yte)
        print(f"epoch {ep} test acc {acc * 100:.2f}%", flush=True)
    np.savez(os.path.join(os.path.dirname(__file__), "model_float.npz"), **P)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
