import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a",help = "a is on")
parser.add_argument("--b",help = "b is on")
args = parser.parse_args()
if args.a:
	print(args.a.split(','))
if args.b:
	print("bbb")