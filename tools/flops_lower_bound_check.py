import math


PAPER_REPORTED_FLOPS = 40_000_000
PAPER_REPORTED_PARAMS = 406_000
PAPER_BATCH_SIZE = 100


LAYERS = [
    {"name": "conv1", "cin": 1, "cout": 32, "length": 8000, "kernel": 32},
    {"name": "conv2", "cin": 32, "cout": 32, "length": 4000, "kernel": 16},
    {"name": "conv3", "cin": 32, "cout": 64, "length": 2000, "kernel": 9},
    {"name": "conv4", "cin": 64, "cout": 64, "length": 1000, "kernel": 6},
    {"name": "conv5", "cin": 64, "cout": 128, "length": 200, "kernel": 3},
    {"name": "conv6", "cin": 128, "cout": 128, "length": 40, "kernel": 3},
    {"name": "conv7", "cin": 128, "cout": 256, "length": 20, "kernel": 3},
]


def conv1d_macs(cin, cout, length, kernel, groups=1):
    return length * cout * (cin // groups) * kernel


def main():
    total_main_macs = 0
    print("Layer lower-bound MACs from Table 2 Conv1D stages")
    print("name   cin  cout  length  kernel  MACs        MACs(M)  exceeds_40M")
    for layer in LAYERS:
        macs = conv1d_macs(layer["cin"], layer["cout"], layer["length"], layer["kernel"])
        total_main_macs += macs
        print(
            f"{layer['name']:<6} {layer['cin']:>4} {layer['cout']:>5} "
            f"{layer['length']:>7} {layer['kernel']:>7} "
            f"{macs:>11} {macs / 1_000_000:>8.2f}  {macs > PAPER_REPORTED_FLOPS}"
        )

    fc_macs = 256 * 10
    total_main_macs += fc_macs
    print()
    print(f"Main Conv1D + classifier lower-bound MACs: {total_main_macs:,} ({total_main_macs / 1_000_000:.2f}M)")
    print(f"Paper reported FLOPs: {PAPER_REPORTED_FLOPS:,} (40M)")
    print(f"Main-only lower-bound / paper 40M: {total_main_macs / PAPER_REPORTED_FLOPS:.2f}x")
    print()
    print("Batch-size hypothesis")
    print(f"Paper params * paper batch size: {PAPER_REPORTED_PARAMS:,} * {PAPER_BATCH_SIZE} = {PAPER_REPORTED_PARAMS * PAPER_BATCH_SIZE:,}")
    print(f"Paper 40M / paper params: {PAPER_REPORTED_FLOPS / PAPER_REPORTED_PARAMS:.2f}")
    print()
    print("Conclusion:")
    print("If Table 2 input length and Conv2 are correct, 40M cannot be standard per-sample Conv1D FLOPs.")
    print("The 40M number is numerically consistent with params * batch size, which is not standard FLOPs.")


if __name__ == "__main__":
    main()
