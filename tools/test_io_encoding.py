#!/usr/bin/env python3
import locale
import os
import sys


envname = "PYTHONIOENCODING"
print("{}:\t{}".format(envname, os.environ.get(envname)))

for set_locale in [False]:
    print("locale({}):\t{}".format(set_locale,
                                   locale.getpreferredencoding(set_locale)))

for streamname in "stdout stderr stdin".split():
    stream = getattr(sys, streamname)
    print("device({}):\t{}".format(streamname,
                                   os.device_encoding(stream.fileno())))
    print("{}.encoding:\t{}".format(streamname, stream.encoding))

for set_locale in [False, True]:
    print("locale({}):\t{}".format(set_locale,
locale.getpreferredencoding(set_locale)))
