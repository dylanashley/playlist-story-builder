#!/usr/bin/env python3
# -*- coding: ascii -*-

from tqdm import tqdm
from typing import Any, Callable, Dict, List
import argparse
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import seaborn as sns
import sys
import tempfile
import zipfile

__version__ = "1.0.0"


def parse_args(args: List[str] = sys.argv[1:]) -> Dict[str, Any]:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="embeds a story into a music playlist by sorting "
        "the playlist so that the order of the music follows a "
        "narrative arc",
        prog="psb",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="individual audio files that make up the playlist",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {}".format(__version__),
    )
    args = vars(parser.parse_args(args))
    for filename in args["files"]:
        assert os.path.isfile(filename)
    return args


def get_value(filename: str) -> float:
    """Returns a narrative arc value for an audio file."""
    loader = essentia.standard.MonoLoader(filename=filename, sampleRate=11025)
    audio = loader()
    value = alg(audio)[0]
    return value


def scale(
    value: float, start_min: float, start_max: float, end_min: float, end_max: float
) -> float:
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min)


def narrative_arc_template(time: float, order: int = 2) -> float:
    """Returns the value of a narrative arc template for a time in the
    range [0, 1].
    """
    assert 0 <= time <= 1
    assert order in [1, 2]
    if order == 1:
        if time <= 0.2:
            return 1 / 2 + 5 / 4 * time
        elif time <= 0.5:
            return 5 / 4 - 5 / 2 * time
        elif time <= 0.8:
            return -5 / 3 + 10 / 3 * time
        else:
            return 2 - 5 / 4 * time
    else:
        if time <= 0.2:
            return 1 / 2 + 5 / 2 * time - 25 / 4 * time ** 2
        elif time <= 0.3:
            return -1 / 4 + 10 * time - 25 * time ** 2
        elif time <= 0.5:
            return 25 / 8 - 25 / 2 * time + 25 / 2 * time ** 2
        elif time <= 0.65:
            return 50 / 9 - 200 / 9 * time + 200 / 9 * time ** 2
        elif time <= 0.8:
            return -119 / 9 + 320 / 9 * time - 200 / 9 * time ** 2
        else:
            return -3 + 10 * time - 25 / 4 * time ** 2


def fit_values(
    values: Dict[str, float],
    template: Callable[[float], float] = narrative_arc_template,
) -> List[str]:
    """Fits a set of narrative arc values to a narrative arc template."""
    values = collections.OrderedDict(values)
    filenames = list(values.keys())
    rv = [-1 for _ in range(len(values))]
    distances = np.zeros((len(values), len(values)), dtype=float)
    for i, y in enumerate(values.values()):
        for j in range(len(values)):
            x = scale(j, 0, len(values), 0, 1)
            distances[i, j] = abs(y - template(x))

    # binary search to find smallest deviation matching
    candidates = np.sort(distances.flatten())
    min_idx = 0
    max_idx = len(candidates) - 1
    while min_idx != max_idx:
        pivot = min_idx + (max_idx - min_idx) // 2
        edges = [
            (filenames[i], j)
            for i in range(len(values))
            for j in range(len(values))
            if distances[i, j] <= candidates[pivot]
        ]
        graph = nx.Graph(edges)
        try:
            matching = nx.bipartite.maximum_matching(graph)
            if all([filename in matching for filename in filenames]):
                max_idx = pivot
            else:
                min_idx = pivot + 1
        except nx.AmbiguousSolution:
            min_idx = pivot + 1

    # clean up smallest deviation matching
    edges = [
        (filenames[i], j, {"weight": distances[i, j]})
        for i in range(len(values))
        for j in range(len(values))
        if distances[i, j] <= candidates[max_idx]
    ]
    graph = nx.Graph(edges)
    matching = nx.bipartite.minimum_weight_full_matching(graph)
    return [matching[i] for i in range(len(filenames))]


def main(args: Dict[str, Any]) -> None:
    if len(args["files"]) == 1:
        print(args["files"][0])
        return
    values = dict()
    for i, filename in enumerate(
        tqdm(args["files"], ascii=True, desc="Extracting Tempos")
    ):
        values[filename] = get_value(filename)
    min_value = min(values.values())
    max_value = max(values.values())
    for k, v in values.items():
        values[k] = scale(v, min_value, max_value, 0, 1)
    playlist = fit_values(values)
    for song in playlist:
        print(song)
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    plt.subplots(1, 1, figsize=(6, 6))
    x = np.linspace(0, 1, num=1000)
    plt.plot(
        [scale(i, 0, 1, 1, len(playlist)) for i in x],
        [scale(narrative_arc_template(i), 0, 1, min_value, max_value) for i in x],
        label="Narrative Arc Template",
    )
    plt.plot(
        np.arange(len(playlist)) + 1,
        [scale(values[song], 0, 1, min_value, max_value) for song in playlist],
        "o-",
        label="Playlist",
    )
    plt.xticks(np.arange(len(playlist)) + 1, playlist, rotation=270)
    plt.legend(loc="upper left", frameon=False)
    plt.ylabel("Tempo (bpm)", labelpad=5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # to avoid tensorflow warnings
    import essentia.standard

    essentia.log.infoActive = False  # to avoid essentia info logging

    with tempfile.TemporaryDirectory() as tempdir:
        alg = essentia.standard.TempoCNN(
            graphFilename=zipfile.ZipFile(
                os.path.join(os.getcwd(), os.path.dirname(__file__))
            ).extract("models/deepsquare-k16-3.pb", path=tempdir)
        )  # to avoid multiple model loads
        main(args)
