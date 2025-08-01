# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import kv

import PRyM.PRyM_init as PRyMini
PRyMini.aTid_flag = True
PRyMini.compute_bckg_flag = True
PRyMini.compute_nTOp_flag = True
PRyMini.compute_nTOp_thermal_flag = False
PRyMini.save_bckg_flag = True
PRyMini.save_nTOp_flag = False
PRyMini.save_nTOp_thermal_flag = False
PRyMini.verbose_flag = False
PRyMini.stasis_flag = True

print(" #################################################")
print(" PRyMordial: Run with small network -- Python call")
print(" #################################################")
PRyMini.smallnet_flag = True
PRyMini.julia_flag = False
start_time = time.time()
import PRyM.test_main3 as PRyMmain
# import PRyM.test_main2 as PRyMmain
# import PRyM.test_main_james as PRyMmain
# import PRyM.new_Main as PRyMmain
res = PRyMmain.PRyMclass().PRyMresults()
print(" ")
print(" Neff --> ",res[0])
print(" Ωνh2 x 10^6 (rel) --> ",res[1])
print(" Σmν/Ωνh2 [eV] --> ",res[2])
print(" YP (CMB) --> ",res[3])
print(" YP (BBN) --> ",res[4])
print(" D/H x 10^5 --> ",res[5])
print(" He3/H x 10^5 --> ",res[6])
print(" Li7/H x 10^10 --> ",res[7])
print(" ")
print("--- running time: %s seconds ---" % (time.time() - start_time))

print(" ")
print(" #################################################")
print(" PRyMordial: Run with large network -- Python call")
print(" #################################################")
PRyMini.smallnet_flag = False
PRyMini.julia_flag = False
start_time = time.time()
# import PRyM.PRyM_main as PRyMmain
res = PRyMmain.PRyMclass().PRyMresults()
print(" ")
print(" Neff --> ",res[0])
print(" Ωνh2 x 10^6 (rel) --> ",res[1])
print(" Σmν/Ωνh2 [eV] --> ",res[2])
print(" YP (CMB) --> ",res[3])
print(" YP (BBN) --> ",res[4])
print(" D/H x 10^5 --> ",res[5])
print(" He3/H x 10^5 --> ",res[6])
print(" Li7/H x 10^10 --> ",res[7])
print(" ")
print("--- running time: %s seconds ---" % (time.time() - start_time))

# ! JULIA RUN -- Commented out for testing right now
# julia_packages_flag = True
# import importlib.util
# package_name_vec = ['julia','diffeqpy']
# for i in range(len(package_name_vec)):
#     spec = importlib.util.find_spec(package_name_vec[i])
#     if spec is None:
#         print(package_name_vec[i] +" is not installed")
#         julia_packages_flag = False
# if(julia_packages_flag):
#     print(" ")
#     print(" ################################################")
#     print(" PRyMordial: Julia initialization (small network)")
#     print(" ################################################")
#     PRyMini.smallnet_flag = True
#     PRyMini.julia_flag = True
#     # import PRyM.PRyM_main as PRyMmain
#     res = PRyMmain.PRyMclass().PRyMresults()

#     print(" ")
#     print(" ################################################")
#     print(" PRyMordial: Run with small network -- Julia call")
#     print(" ################################################")
#     PRyMini.smallnet_flag = True
#     PRyMini.julia_flag = True
#     start_time = time.time()
#     # import PRyM.PRyM_main as PRyMmain
#     res = PRyMmain.PRyMclass().PRyMresults()
#     print(" ")
#     print(" Neff --> ",res[0])
#     print(" Ωνh2 x 10^6 (rel) --> ",res[1])
#     print(" Σmν/Ωνh2 [eV] --> ",res[2])
#     print(" YP (CMB) --> ",res[3])
#     print(" YP (BBN) --> ",res[4])
#     print(" D/H x 10^5 --> ",res[5])
#     print(" He3/H x 10^5 --> ",res[6])
#     print(" Li7/H x 10^10 --> ",res[7])
#     print(" ")
#     print("--- running time: %s seconds ---" % (time.time() - start_time))

#     print(" ")
#     print(" ################################################")
#     print(" PRyMordial: Julia initialization (large network)")
#     print(" ################################################")
#     PRyMini.smallnet_flag = False
#     PRyMini.julia_flag = True
#     # import PRyM.PRyM_main as PRyMmain
#     res = PRyMmain.PRyMclass().PRyMresults()

#     print(" ")
#     print(" ################################################")
#     print(" PRyMordial: Run with large network -- Julia call")
#     print(" ################################################")
#     PRyMini.smallnet_flag = False
#     PRyMini.julia_flag = True
#     start_time = time.time()
#     # import PRyM.PRyM_main as PRyMmain
#     res = PRyMmain.PRyMclass().PRyMresults()
#     print(" ")
#     print(" Neff --> ",res[0])
#     print(" Ωνh2 x 10^6 (rel) --> ",res[1])
#     print(" Σmν/Ωνh2 [eV] --> ",res[2])
#     print(" YP (CMB) --> ",res[3])
#     print(" YP (BBN) --> ",res[4])
#     print(" D/H x 10^5 --> ",res[5])
#     print(" He3/H x 10^5 --> ",res[6])
#     print(" Li7/H x 10^10 --> ",res[7])
#     print(" ")
#     print("--- running time: %s seconds ---" % (time.time() - start_time))


#!###########
# A correct output should look like this:
#
#
#
#  #################################################
#  PRyMordial: Run with small network -- Python call
#  #################################################
# WARNING: import of MainInclude.eval into Main conflicts with an existing identifier; ignored.
# WARNING: could not import MainInclude.include into Main
 
#  Neff -->  3.0443885202772574
#  Ωνh2 x 10^6 (rel) -->  5.699407167710109
#  Σmν/Ωνh2 [eV] -->  93.03360797907557
#  YP (CMB) -->  0.24555567642743933
#  YP (BBN) -->  0.24688155978280088
#  D/H x 10^5 -->  2.4551722192464647
#  He3/H x 10^5 -->  1.0413418238433338
#  Li7/H x 10^10 -->  5.4967567730531215
 
# --- running time: 13.096626043319702 seconds ---
 
#  #################################################
#  PRyMordial: Run with large network -- Python call
#  #################################################
 
#  Neff -->  3.0443885202772574
#  Ωνh2 x 10^6 (rel) -->  5.699407167710109
#  Σmν/Ωνh2 [eV] -->  93.03360797907557
#  YP (CMB) -->  0.2455609875736719
#  YP (BBN) -->  0.24688689022211488
#  D/H x 10^5 -->  2.4605778244170327
#  He3/H x 10^5 -->  1.0419820734789897
#  Li7/H x 10^10 -->  5.424599835186072
 
# --- running time: 5.842180967330933 seconds ---
