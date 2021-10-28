# playlist-story-builder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project attempts to embed a story into a music playlist by sorting the playlist so that the order of the music follows a narrative arc. Currently, music is fitted to a fixed narrative arc template based on an estimate of the tempo of the songs in beats per minute.

## Installation

This project is implemented in [Python](https://www.python.org/) and uses [TensorFlow](https://www.tensorflow.org/) models from the [Essentia](https://essentia.upf.edu/) library to extract the tempo of the songs.

To use this project, first follow the [Essentia installation instructions](https://essentia.upf.edu/installing.html) to install it with TensorFlow support. Then install the remaining required packages using pip:
```bash
pip install -r requirements.txt
```

Afterwards, you can use the makefile to compile and then install an executable Python zip archive:
```bash
make all
sudo make install
```

To run the program, execute it while passing the audio files as command-line arguments:
```bash
psb files [files ...] >> playlist.txt
```

## Acknowledgements

The algorithm to fit the calculated narrative arc values to a narrative arc template was originally proposed by Dr. Zachary Friggstad.
