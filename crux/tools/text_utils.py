import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import yaml
import argparse
import json

import numpy as np
from tqdm import tqdm

from prompts.mds import *

def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

