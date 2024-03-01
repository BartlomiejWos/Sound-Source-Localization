#!/usr/bin/env python3

import sys
import socket
import argparse
import queue

import numpy as np
import sounddevice as sd


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--channels", type=int, default=1,
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=16000,
    )
    parser.add_argument(
        "-b", "--blocksize", type=int, default=64,
    )
    parser.add_argument(
        "-d", "--device", type=int, default=2,
    )
    parser.add_argument(
        "-i", "--ip", type=str, default="0.0.0.0",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=9999,
    )

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    # Get params
    channels = args.channels
    fs = args.frequency
    blocksize = args.blocksize
    device = args.device
    dtype = np.int16

    IP = args.ip
    PORT = args.port
    addr = (IP, PORT)


    bytes_per_sample = np.dtype(dtype).itemsize
    print("Frame length:\t\t", blocksize/fs * 1000, "[ms]")
    print("Bytes per frame:\t", channels * blocksize * bytes_per_sample)



    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        q.put(indata.copy())

    audio_stream = sd.InputStream(
        samplerate=fs,
        blocksize=blocksize,
        channels=channels,
        dtype=dtype,
        callback=audio_callback,
        device=device
    )


    try:
        with audio_stream:
            sockd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            while True:
                block = q.get().T.tobytes("C")
                sockd.sendto(block, addr)
    except KeyboardInterrupt:
        pass