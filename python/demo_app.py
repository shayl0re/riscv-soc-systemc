"""Live demo: draw a digit -> the SystemC RISC-V SoC classifies it.
Run:  python3 demo_app.py          (needs Tkinter; on WSL2/Windows 11 a window opens via WSLg)
      python3 demo_app.py --cli    (no GUI: classifies test image #N through the SoC)
"""
import os, re, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(HERE, "..", "systemc")


def run_soc(img28, pe=128, pipelined=False, wrong_key=False, privacy=False, hardened=False):
    """img28: 28x28 uint8 array. returns dict with result text from the SoC simulation."""
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False); f.write(np.asarray(img28, np.uint8).tobytes()); f.close()
    scen = "wrongkey" if wrong_key else ("privacy" if privacy else "normal")
    cmd = ["./soc_sim", "--image", f.name, "--scenario", scen, "--pe", str(pe), "--layers"] + (["--pipe"] if pipelined else []) + (["--hardened"] if hardened else [])
    out = subprocess.run(cmd, cwd=SC, capture_output=True, text=True).stdout
    os.unlink(f.name)
    r = {"raw": out, "denied": "AUTH DENIED" in out}
    m = re.search(r"result=\s*(\d+) \(label \d+, clean prediction (\d+)\)\s+accel (\d+) cycles = ([\d.]+) us", out)
    if m: r.update(result=int(m[1]), clean=int(m[2]), cycles=int(m[3]), us=float(m[4]))
    return r


def preprocess(canvas_img):
    """canvas_img: 2D float array 0..255 (white ink on black). Crop, resize to 20x20, centre in 28x28 like MNIST."""
    from PIL import Image
    a = np.asarray(canvas_img, np.float32)
    ys, xs = np.where(a > 30)
    if len(ys) == 0: return np.zeros((28, 28), np.uint8)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape; s = 20.0 / max(h, w)
    im = Image.fromarray(a.astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = np.zeros((28, 28), np.float32); b = np.asarray(im, np.float32)
    out[(28 - b.shape[0]) // 2:(28 - b.shape[0]) // 2 + b.shape[0], (28 - b.shape[1]) // 2:(28 - b.shape[1]) // 2 + b.shape[1]] = b
    ys, xs = np.nonzero(out); cy, cx = (out * np.arange(28)[:, None]).sum() / out.sum(), (out * np.arange(28)[None, :]).sum() / out.sum()
    out = np.roll(np.roll(out, int(round(13.5 - cy)), 0), int(round(13.5 - cx)), 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def gui():
    import tkinter as tk
    from PIL import Image, ImageDraw
    S = 280; root = tk.Tk(); root.title("Secure Edge-AI RISC-V SoC (SystemC simulation)")
    cv = tk.Canvas(root, width=S, height=S, bg="black", cursor="pencil"); cv.grid(row=0, column=0, rowspan=8, padx=8, pady=8)
    pil = Image.new("L", (S, S), 0); dr = ImageDraw.Draw(pil); last = [None]
    def draw(e):
        if last[0]: cv.create_line(*last[0], e.x, e.y, fill="white", width=18, capstyle="round"); dr.line([*last[0], e.x, e.y], fill=255, width=18)
        last[0] = (e.x, e.y)
    cv.bind("<B1-Motion>", draw); cv.bind("<ButtonRelease-1>", lambda e: last.__setitem__(0, None))
    pe = tk.IntVar(value=128); wk = tk.BooleanVar(); pv = tk.BooleanVar(); pp = tk.BooleanVar(); hd = tk.BooleanVar()
    tk.Label(root, text="PEs:").grid(row=0, column=1, sticky="w")
    f = tk.Frame(root); f.grid(row=0, column=2, sticky="w")
    for v in (1, 9, 128): tk.Radiobutton(f, text=str(v), variable=pe, value=v).pack(side="left")
    tk.Checkbutton(root, text="pipelined", variable=pp).grid(row=1, column=1, columnspan=2, sticky="w")
    tk.Checkbutton(root, text="WRONG secret key (attack)", variable=wk).grid(row=2, column=1, columnspan=2, sticky="w")
    tk.Checkbutton(root, text="privacy mode (LFSR noise)", variable=pv).grid(row=3, column=1, columnspan=2, sticky="w")
    tk.Checkbutton(root, text="hardened design (keyed mask)", variable=hd).grid(row=4, column=1, columnspan=2, sticky="w")
    out = tk.Label(root, text="Draw a digit, then press Classify", font=("Arial", 14), justify="left", wraplength=300); out.grid(row=6, column=1, columnspan=2, padx=8)
    def classify():
        r = run_soc(preprocess(np.asarray(pil)), pe.get(), pp.get(), wk.get(), pv.get(), hd.get())
        if r["denied"]: out.config(text="AUTHENTICATION FAILED\nCNN accelerator stayed idle.", fg="red")
        elif "result" in r:
            t = f"SoC output: {r['result']}\n(accelerator latency {r['us'] / 1000:.3f} ms, {r['cycles']} cycles)"
            if pv.get(): t += f"\nprivacy ON - clean class would be {r['clean']}"
            out.config(text=t, fg="green" if not pv.get() else "orange")
        else: out.config(text="simulation error:\n" + r["raw"][-200:], fg="red")
    def clear(): cv.delete("all"); dr.rectangle([0, 0, S, S], fill=0); out.config(text="Draw a digit, then press Classify", fg="black")
    tk.Button(root, text="Classify on RISC-V SoC", command=classify).grid(row=5, column=1, columnspan=2, pady=6)
    tk.Button(root, text="Clear", command=clear).grid(row=7, column=1, columnspan=2)
    root.mainloop()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        k = int(sys.argv[sys.argv.index("--cli") + 1]) if len(sys.argv) > sys.argv.index("--cli") + 1 else 0
        im = np.fromfile(os.path.join(SC, "data", "test_images.bin"), np.uint8)[784 * k:784 * (k + 1)].reshape(28, 28)
        for kw in [{}, {"wrong_key": True}, {"privacy": True}]:
            r = run_soc(im, **kw); print(kw or "normal", "->", "DENIED" if r["denied"] else f"result={r.get('result')}  ({r.get('us', 0)/1000:.3f} ms)")
    else:
        gui()
