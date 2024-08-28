#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import json
import re

def main(argv):
    if len(argv) == 2:
        js = json.loads(argv[1])
    else:
        js = json.loads(argv[3])
    sys.exit(js['status'])

if __name__ == '__main__':
    main(sys.argv)
