import argparse

from openk4a.playback import OpenK4APlayback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input MKV file.")
    args = parser.parse_args()

    azure = OpenK4APlayback(args.input)
    azure.open()

    for stream in azure.streams:
        print(stream)

    azure.close()


if __name__ == "__main__":
    main()
