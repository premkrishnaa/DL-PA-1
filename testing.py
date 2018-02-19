import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a",help = "a is on")
parser.add_argument("--b",help = "b is on")
parser.add_argument('--feature', dest='feature', action='store_true')
parser.add_argument('--no-feature', dest='feature', action='store_false')
parser.set_defaults(feature=True)

args = parser.parse_args()
if args.a:
	print(args.a.split(','))
if args.b:
	print("bbb")

print args.feature