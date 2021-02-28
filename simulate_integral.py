import numpy as np
import scipy.special
import matplotlib.pyplot as pplot


scipy.special.beta(5,5)
scipy.special.betainc



def getBeta(a,b):
    return scipy.special.beta(a,b)

def getIncbeta(x,a,b):
    return scipy.special.betainc(a,b,x) #* getBeta(a,b)


def getIncbetaval(x,a,b):
    pass

def getBetaPDF(x,a,b):
    return (np.power(x, a-1) * (1-x) ** (b-1)) / getBeta(a,b)


def getIntegrandVals(x,a_l,b_l,a_b,b_b, M):
    return (getBetaPDF(x,a_l,b_l) * (getIncbeta(x,a_l,b_l)) **(M-2)) * (1-getIncbeta(x,a_b,b_b))


def computeIntegral(M, a_l,b_l,a_b,b_b):
    x_axis = np.linspace(0,1,1000)
    delta_x = 1/1000
    # x_axis=0.5
    func_value =getIntegrandVals(x_axis, a_l,b_l,a_b,b_b, M)
    # func_value = x_axis ** 2
    return np.sum(delta_x*func_value) * (M-1)



def _run():
    M = 100
    a_l = 5
    b_l = 5
    a_b = 6
    b_b = 4
    _a_b = np.arange(5, 10, 0.2)
    _b_b = np.arange(5,0, -0.2)

    ret = []
    for (a_b,b_b) in zip(_a_b,_b_b):
        ret.append(computeIntegral(M,a_l,b_l,a_b,b_b))

    pplot.plot(_a_b,ret, label="k=100")
    M = 50
    ret = []
    for (a_b, b_b) in zip(_a_b, _b_b):
        ret.append(computeIntegral(M, a_l, b_l, a_b, b_b))
    pplot.plot(_a_b,ret, label="k=50")
    M= 10
    ret = []
    for (a_b, b_b) in zip(_a_b, _b_b):
        ret.append(computeIntegral(M, a_l, b_l, a_b, b_b))


    print(ret)
    pplot.plot(_a_b,ret, label="k=10")
    # pplot.xlim(0,1)
    # pplot.xticks(_a_b)
    pplot.legend()
    pplot.xlabel("alpha values")
    pplot.ylabel("Probability")
    pplot.xticks(np.arange(5,10,0.5))
    pplot.show()

if __name__ == '__main__':
    _run()