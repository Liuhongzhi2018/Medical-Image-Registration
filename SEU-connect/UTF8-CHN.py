#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys

# convert utf-8 string to CHN string
def main(argv):
    sys.stdout.write(re.sub(r'\\u\w{4}',
                    lambda e: unichr(int(e.group(0)[2:], 16)).encode('utf-8'),
                    argv[1]))

if __name__ == '__main__':
    main(sys.argv)
