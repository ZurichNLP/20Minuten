#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

dir_path = Path(sys.argv[1])


dates = Counter()

for file_path in dir_path.iterdir():
    dates[(datetime.fromtimestamp(os.stat(file_path).st_mtime).date()).isoformat()] += 1

if len(dates) > 1:
    print('[!] files have different modfiction dates!')
    print(dates)

else:
    print(f'All files in {dir_path} have the same modification date')
    print(dates)