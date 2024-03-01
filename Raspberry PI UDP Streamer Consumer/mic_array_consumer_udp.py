#!/usr/bin/env python

import argparse
import socket
import queue

import numpy as np
import sounddevice as sd


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--frequency', type=int, default=16000,
    )

    parser.add_argument(
        '-b', '--blocksize', type=int, default=64,
    )

    parser.add_argument(
        '-c', '--channels', type=int, default=1,
    )

    parser.add_argument(
        '--ip', type=str, default="0.0.0.0",
    )

    parser.add_argument(
        '--port', type=int, default=9999,
    )

    parser.add_argument(
        '--time', type=int, default=250,
    )

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    sample_dtype = np.int16
    # audio
    fs = args.frequency
    channels = args.channels
    blocksize = args.blocksize

    bytes_per_sample = np.dtype(sample_dtype).itemsize

    bpf = channels * blocksize * bytes_per_sample

    print("Bytes per packet:", bpf)
    # communication
    IP = args.ip
    PORT = args.port
    addr = (IP, PORT)

    q = queue.Queue()

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = q.get().copy().reshape((blocksize, 1))

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sockd:
        sockd.bind(addr)


        stream = sd.Stream(
                samplerate=fs,
                channels=1,
                blocksize=blocksize,
                callback=callback,
                dtype=np.int16
        )

        with stream:
            try:
                while True:
                    block, _ = sockd.recvfrom(bpf)
                    if not block:
                        break
                    while len(block) < bpf:
                        foo = sockd.recv(bpf - len(block))
                        block = block + foo
                    block = np.frombuffer(block, sample_dtype)
                    block = block.reshape((channels, blocksize))
                    beam = np.mean(block, axis=0)
                    q.put(beam)
            except KeyboardInterrupt:
                pass