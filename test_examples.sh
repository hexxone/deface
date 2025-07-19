#!/bin/bash

# Script that tests black-box tests deface with examples.
# Before running, install deface and cd to the repo root


# Show help, checking if the main script runs without errors
deface --help

# Create a temporary directory for outputs
tmpdir=$(mktemp -d -t deface-XXXXXXXXXX)

# Test deface with the example image, write output to temporary directory
deface examples/original/city.jpg -o ${tmpdir}/city_anonymized.jpg

# Test deface with the example video, write output to temporary directory
python3 examples/create_city_video.py
deface examples/city.mp4 -o ${tmpdir}/test_video_anonymized.mp4
