from config.Config import *
from models import TransE
import json
import os
con = Config()
con.set_use_gpu(False)
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("./benchmarks/homework4/")
#True: Input test files from the same folder.
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_test_model(TransE)
con.test()