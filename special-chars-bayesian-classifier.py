import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from scipy.ndimage import median_filter
from skimage.measure import euler_number

folder = Path("special chars jpeg")
classes = ["#", "$", "&", "*", "@"]
K = len(classes)
all_pictures = sorted(folder.rglob("*.jp*g"))


def preprocessing(path):
    img = io.imread(path, as_gray=True)

    # 1. binarization
    M, N = img.shape
    T = 0.8
    m = np.percentile(img, 95)
    imgbin = (img > T * m).astype(int)

    # 2. first crop: remove the dark outer frame
    left = 0
    while left < N - 1 and np.sum(imgbin[:, left]) < T * M:
        left += 1

    right = N - 1
    while right > 0 and np.sum(imgbin[:, right]) < T * M:
        right -= 1

    up = 0
    while up < M - 1 and np.sum(imgbin[up, :]) < T * N:
        up += 1

    down = M - 1
    while down > 0 and np.sum(imgbin[down, :]) < T * N:
        down -= 1

    imgbin = imgbin[up:down, left:right]

    # 3. median filter 3x3
    imgbin = median_filter(imgbin, size=(3, 3))

    # 4. second crop: remove the white margins around the symbol
    P = 0.99
    M, N = imgbin.shape

    left = 0
    while left < N - 1 and np.sum(imgbin[:, left]) > P * M:
        left += 1

    right = N - 1
    while right > 0 and np.sum(imgbin[:, right]) > P * M:
        right -= 1

    up = 0
    while up < M - 1 and np.sum(imgbin[up, :]) > P * N:
        up += 1

    down = M - 1
    while down > 0 and np.sum(imgbin[down, :]) > P * N:
        down -= 1

    imgbin = imgbin[up:down, left:right]

    return imgbin  # 0 = symbol, 1 = background


feats = ["centre", "sym_horiz", "sym_vert", "crossings",
         "left", "right", "top", "bottom", "ink_ratio", "euler", "aspect_ratio"]


def feature_extraction(img):
    ink = (img == 0)
    H, W = ink.shape
    n = ink.sum() + 1e-9
    mh = max(1, int(0.1 * H))
    mw = max(1, int(0.1 * W))

    centre = ink[H // 4:3 * H // 4, W // 4:3 * W // 4].sum() / n
    sym_horiz = (ink == ink[:, ::-1]).mean()
    sym_vert = (ink == ink[::-1, :]).mean()
    crossings = np.sum(np.diff(ink[:, W // 2].astype(int)) == 1)
    left = ink[:, :mw].sum() / n
    right = ink[:, -mw:].sum() / n
    top = ink[:mh, :].sum() / n
    bottom = ink[-mh:, :].sum() / n
    ink_ratio = ink.mean()
    euler = euler_number(ink, connectivity=2)
    aspect_ratio = H / W
    return [centre, sym_horiz, sym_vert, crossings,
            left, right, top, bottom, ink_ratio, euler, aspect_ratio]


F = []
y = []
paths = []

for pic in all_pictures:
    img_processed = preprocessing(pic)
    feat = feature_extraction(img_processed)
    F.append(feat)
    y.append(classes.index(pic.parent.name))
    paths.append(pic)

F = np.array(F)
y = np.array(y)


def divide(X):
    np.random.seed(42)
    idX = np.random.permutation(len(X))
    limit = int(len(X) * 0.8)
    return X[idX[:limit]], X[idX[limit:]]


X_train = []
X_test = []
y_train = []
y_test = []

for c in range(K):
    Xc_train, Xc_test = divide(F[y == c])
    X_train.append(Xc_train)
    X_test.append(Xc_test)
    y_train += [c] * len(Xc_train)
    y_test += [c] * len(Xc_test)

X_train = np.concatenate(X_train)
X_test = np.concatenate(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# standardization (mean and std from the train set only)
mu_all = X_train.mean(axis=0)
sd_all = X_train.std(axis=0) + 1e-9
X_train = (X_train - mu_all) / sd_all
X_test = (X_test - mu_all) / sd_all

D = X_train.shape[1]
reg = 1e-2 * np.eye(D)
total_tr = len(y_train)

mu = []
inv_Sigma = []
logdet = []
log_prior = []

for c in range(K):
    Xc = X_train[y_train == c]
    Sigma = np.cov(Xc.T) + reg
    mu.append(np.mean(Xc, axis=0))
    inv_Sigma.append(np.linalg.inv(Sigma))
    logdet.append(np.linalg.slogdet(Sigma)[1])
    log_prior.append(np.log(len(Xc) / total_tr))


def discriminant_function(x, mu, inv_Sigma, logdet, log_prior):
    diff = x - mu
    return -0.5 * logdet - 0.5 * (diff @ inv_Sigma @ diff) + log_prior


def predict(X):
    preds = []
    for x in X:
        g = []
        for c in range(K):
            g.append(discriminant_function(x, mu[c], inv_Sigma[c], logdet[c], log_prior[c]))
        preds.append(np.argmax(g))
    return np.array(preds)


y_train_pred = predict(X_train)
y_test_pred = predict(X_test)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((K, K), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"Train accuracy: {np.mean(y_train == y_train_pred) * 100:.2f}%")
print(f"Test accuracy:  {np.mean(y_test == y_test_pred) * 100:.2f}%")

# ---------------- 1. Preprocessed Symbols ----------------
fig, axes = plt.subplots(2, K, figsize=(11, 4.5))
for c in range(K):
    sample_pic = paths[np.where(y == c)[0][0]]
    axes[0, c].imshow(io.imread(sample_pic), cmap="gray")
    axes[0, c].set_title("Raw: " + classes[c])
    axes[0, c].axis("off")

    axes[1, c].imshow(preprocessing(sample_pic), cmap="gray")
    axes[1, c].set_title("Cropped: " + classes[c])
    axes[1, c].axis("off")

plt.tight_layout()
plt.savefig("preprocessing.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- 2. 2D Decision Boundaries ----------------
i, j = 8, 10  # ink_ratio vs aspect_ratio
X_tr2 = X_train[:, [i, j]]
X_te2 = X_test[:, [i, j]]

mu2 = []
inv2 = []
ld2 = []
reg2 = 1e-2 * np.eye(2)

for c in range(K):
    Xc2 = X_tr2[y_train == c]
    S2 = np.cov(Xc2.T) + reg2
    mu2.append(np.mean(Xc2, axis=0))
    inv2.append(np.linalg.inv(S2))
    ld2.append(np.linalg.slogdet(S2)[1])

xx, yy = np.meshgrid(
    np.linspace(X_te2[:, 0].min() - 0.5, X_te2[:, 0].max() + 0.5, 250),
    np.linspace(X_te2[:, 1].min() - 0.5, X_te2[:, 1].max() + 0.5, 250),
)

Z = []
for pt in np.c_[xx.ravel(), yy.ravel()]:
    g = []
    for c in range(K):
        g.append(discriminant_function(pt, mu2[c], inv2[c], ld2[c], log_prior[c]))
    Z.append(np.argmax(g))
Z = np.array(Z).reshape(xx.shape)

bg_colors = ["#c7e9c0", "#c6dbef", "#dadaeb", "#fcc5c0", "#fee6ce"]
colors = ["#1b7837", "#2171b5", "#6a51a3", "#dd3497", "#d95f02"]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], colors=bg_colors, alpha=0.7)
plt.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5, 3.5], colors="#333333", linewidths=1.2)

for c in range(K):
    plt.scatter(X_te2[y_test == c, 0], X_te2[y_test == c, 1], c=colors[c], label=classes[c], edgecolors="white", s=45)
    plt.scatter(mu2[c][0], mu2[c][1], c="gold", marker="*", s=240, edgecolors="black", linewidths=1.1)

plt.xlabel(feats[i], fontsize=11)
plt.ylabel(feats[j], fontsize=11)
plt.title(f"Decision Boundaries ({feats[i]} vs {feats[j]})", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("decision_boundaries.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- 3. Confusion Matrices ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cm, title in zip(axes, [cm_train, cm_test], ["Train data", "Test data"]):
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for a in range(K):
        for b in range(K):
            col = "white" if cm[a, b] > cm.max() / 2 else "black"
            ax.text(b, a, int(cm[a, b]), ha="center", va="center", color=col, fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------- 4. Misclassified Examples ----------------
F_std = (F - mu_all) / sd_all
all_preds = predict(F_std)
wrong = np.where(all_preds != y)[0][:4]

if len(wrong) > 0:
    fig, axes = plt.subplots(2, len(wrong), figsize=(10, 5), squeeze=False)
    for k, idx in enumerate(wrong):
        axes[0, k].imshow(io.imread(paths[idx]), cmap="gray")
        axes[0, k].set_title(f"True: {classes[y[idx]]}\nPred: {classes[all_preds[idx]]}", color="crimson", fontweight="bold")
        axes[0, k].axis("off")

        axes[1, k].imshow(preprocessing(paths[idx]), cmap="gray")
        axes[1, k].axis("off")
    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=200, bbox_inches="tight")
    plt.show()