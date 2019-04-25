import pickle

unit_test = pickle.load(open("./../unit_test.pkl",'rb'))
print(unit_test)
data = pickle.load(open("./../result.pkl",'rb'))
print(data)
if data==unit_test:
    print("Unit test PASSED",)
else:
    print("FAIL test 1")