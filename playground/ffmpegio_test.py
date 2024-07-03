import time

import ffmpegio as ff

input_path = "test.mkv"
streams = ["v:0", "v:1", "v:2"]

i = 0
with ff.open(str(input_path), "rvv", map=streams, blocksize=1, stream_loop_in=0) as fin:
    while True:
        F = fin.read()
        print(f"{i} = {list(F.keys())}")
        # time.sleep(0.1)
        i += 1
