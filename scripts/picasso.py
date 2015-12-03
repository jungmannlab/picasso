import sys
import subprocess


subprocess.call(['python', '-m' 'picasso'] + sys.argv[1:])
