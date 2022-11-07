#!/usr/bin/env python
"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""
import sys


# if len(sys.argv[1:]) != 2:
#  sys.exit(__doc__)

inner = "D:\\EM0122\\Encoding\\Behavior\\RawBehavioralData\\ENCODING_EM0122_10_of_20_StimulusInfo.pkl"
outer = "D:\\EM0122\\Encoding\\Behavior\\RawBehavioralData\\ENCODING_EM0122_10_of_20_StimulusInfo_conv.pkl"

content = ''
outsize = 0
with open(inner, 'rb') as infile:
  content = infile.read()
with open(outer, 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + '\n')

print("Done. Saved %s bytes." % (len(content)-outsize))

