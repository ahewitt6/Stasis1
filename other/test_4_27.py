import sys
# sys.path.insert(0, 'PRyM/')   # or wherever your .py files live

import PRyM.PRyM_init as PRyMini
# 1) import both classes
# from test_main import PRyMclass as TestPRyM
# from new_Main import PRyMclass as NewPRyM
import PRyM.test_main as TestPRyM
import PRyM.new_Main as NewPRyM

# 2) instantiate with identical config
#    (you must set all the same PRyMini flags so that both skip Julia
#     and go down the pure-Python MT branch)
test = TestPRyM.PRyMclass()
new  = NewPRyM.PRyMclass()

# force both to go into their MT eras:
test.PRyMini.smallnet_flag = False
test.PRyMini.julia_flag    = False
new.PRyMini.smallnet_flag  = False
new.PRyMini.julia_flag     = False

# 3) extract the time-interval and initial vector they both compute
#    (you may need to copy how your __init__ sets t_init, t_fin, Yi_vec)
t_init = test.t_init
t_fin   = test.t_fin
Yi_vec  = test.Yi_vec  # assuming you stored it on self

# 4) grab the right-hand side functions from each
test_f = test.Y_prime_MT    # or however you’ve exposed it
new_f  = new.Y_prime_MT

# 5) evaluate them at the same sample point
test_dY = test_f(t_init, Yi_vec)
new_dY  = new_f( t_init, Yi_vec)

# 6) compare element-by-element
import pandas as pd
df = pd.DataFrame({
    'species': range(len(test_dY)),
    'test_main': test_dY,
    'new_Main':  new_dY,
    'diff':      test_dY - new_dY
})
print(df)
