import numpy as np
import pandas as pd
from pathlib import Path
import os


def apply_model(model, params, batch, batch_size) -> tuple:
    mono_dc2, dipo_dc2 = model.apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    return mono_dc2, dipo_dc2


def flatten(xss):
    return [x for xs in xss for x in xs]


def clip_colors(c):
    return np.clip(c, -0.015, 0.015)


def reshape_dipole(dipo, nDCM):
    d = dipo.reshape(1, 60, 3, nDCM)
    d = np.moveaxis(d, -1, -2)
    d = d.reshape(1, 60 * nDCM, 3)
    return d


def process_df(errors):
    h2kcal = 627.509
    df = pd.DataFrame(flatten(errors))
    df["model"] = df[0].apply(lambda x: np.sqrt(x) * h2kcal)
    df["mono"] = df[1].apply(lambda x: np.sqrt(x) * h2kcal)
    df["dif"] = df["model"] - df["mono"]
    return df


def get_lowest_loss(path, df=False):
    paths = []
    losses = []
    for _ in Path(path).glob("*.pkl"):
        loss = float((_.stem).split("-")[1])
        paths.append(_)
        losses.append(loss)
    if df:
        ans = pd.DataFrame([paths, losses]).T.sort_values(1)
        print(ans)
        return ans
    else:
        ans = pd.DataFrame([paths, losses]).T.sort_values(1).iloc[0][0]
        print(ans)
        return ans


def safe_mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
