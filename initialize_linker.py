import os, sys
sys.path.append(os.path.abspath('.'))
from util.argparser import LinkerArguments, CorpusArguments
from transformers import HfArgumentParser
from time import time

parser = HfArgumentParser((LinkerArguments, CorpusArguments))
args = parser.parse_args()
from linker.faiss_linker import EntityLinkerFaiss    
linker = EntityLinkerFaiss(args)
t0 = time()
linker.insert_dictionary()
t1 = (time() - t0) / 60
print(f'End, uses time {t1} minutes')