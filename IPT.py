# IPT11
# @autor Lukáš Plevač
# @date 17.12.2023

import numpy as np
import math

np.set_printoptions(suppress=True)

###
# LINEAR
##

## Pomocí regresní přímky popište závislost Yi na Xi
# @param Xi np.array of values
# @param Yi np.array of values
# @return beta0 and beta1
def regreseLinearBetas(Xi, Yi):
    # n * beta0 + Xi.sum() * beta1 = Yi.sum()
    # Xi.sum() * beta0 + Xi.pow(2).sum() = Xi.sum * Yi.sum()

    if (Xi.shape != Yi.shape):
        print("error Xi shape is not smae as Yi shape")
        return 0, 0

    Xi_sum      = Xi.sum()

    Xi_pow2_sum = np.power(Xi, 2)
    Xi_pow2_sum = Xi_pow2_sum.sum()
    
    Yi_sum      = Yi.sum()

    n           = Xi.size

    Xi_mul_Yi_sum = (Xi * Yi).sum()


    print("{}Beta0 + {}beta1 = {}".format(n, Xi_sum, Yi_sum))
    print("{}Beta0 + {}beta1 = {}".format(Xi_sum, Xi_pow2_sum, Xi_mul_Yi_sum))

    # clac betas
    xx = np.array([
        [n, Xi_sum],
        [Xi_sum, Xi_pow2_sum]
    ])

    xy = np.array([
        [Yi_sum],
        [Xi_mul_Yi_sum]
    ])

    betas = np.matmul(np.linalg.inv(xx), xy)

    print("#############################")
    print("betas = (X'X)**-1 . X'Y = ")
    print(np.linalg.inv(xx))
    print(".")
    print(xy)
    print("=")
    print(betas)
    print("Beta0 is {}, Beta1 is {}".format(betas[0][0], betas[1][0]))
    print("#############################")

    return betas[0][0], betas[1][0]

## Provede předpověď Y pro X
# @param beta0
# @param beta1
# @param X
# @retrn Predicted value
def regreseLinearPredict(beta0, beta1, x):
    y = beta0 + (beta1 * x)
    print("#############################")
    print("Y = beta0 + beta1 * x = {}".format(y))
    print("#############################")
    return y

# Get Se of regreseLinear
# @param beta0
# @param beta1
# @param Xi
# @param Yi
# @retrn Se
def regreseLinearGetSe(beta0, beta1, Xi, Yi):

    Yi_pow2_sum = np.power(Yi, 2)
    Yi_pow2_sum = Yi_pow2_sum.sum()

    Yi_sum      = Yi.sum()

    Xi_mul_Yi_sum = (Xi * Yi).sum()

    print("Yi.pow(2).sum is {}".format(Yi_pow2_sum))
    print("Yi.sum is {}".format(Yi_pow2_sum))
    print("(Xi * Yi).sum() is {}".format(Xi_mul_Yi_sum))

    Se = Yi_pow2_sum - beta0 * Yi_sum - beta1 * Xi_mul_Yi_sum

    print("#####################")
    print("Se = {} - {} * {} - {} * {} = {}".format(
         Yi_pow2_sum,
         beta0,
         Yi_sum,
         beta1,
         Xi_mul_Yi_sum,
         Se
    ))
    print("#####################")

    return Se

# Get SigmaPow2 regreseLinear
# @param beta0
# @param beta1
# @param Xi
# @param Yi
# @retrn Sigma**2
def regreseLinearGetSigmaPow2(beta0, beta1, Xi, Yi):
    Se = regreseLinearGetSe(beta0, beta1, Xi, Yi)
    n  = Xi.size

    SigmaPow2 = Se / (n-2)

    print("#####################")
    print("SigmaPow2 = s**2 = Se / (n-2) = {} / ({} - 2) = {}".format(
         Se,
         n,
         SigmaPow2
    ))
    print("#####################")

    return SigmaPow2

# Get Qvality or regreseLinear in %
# @param beta0
# @param beta1
# @param Xi
# @param Yi
# @retrn R**2
def regreseLinearGetQvality(beta0, beta1, Xi, Yi):
    prum = np.mean(Yi)

    Sypow2 = (1 / (Yi.size - 1)) * np.power(Yi - prum, 2).sum()

    print("#####################")
    print("Sy**2 = (1 / ( n - 1)) * suma((Yi - prum(Y)) ** 2) = {}".format(
         Sypow2
    ))
    print("#####################")

    St = Sypow2 * (Yi.size - 1)

    print("#####################")
    print("St = ( n - 1) * Sy**2 = {}".format(
         St
    ))
    print("#####################")

    Se = regreseLinearGetSe(beta0, beta1, Xi, Yi)


    Rpow2 = 1 - Se / St

    print("#####################")
    print("R**2 = 1 - Se / St = 1 - {} / {} = {}".format(
         Se,
         St,
         Rpow2
    ))
    print("#####################")

    return Rpow2

###
# HYPERBOLA
##

## Pomocí regresní hyperboly popište závislost Yi na Xi
# @param Xi np.array of values
# @param Yi np.array of values
# @return beta0, beta1 and beta2
def regreseHyperbolaBetas(Xi, Yi):
    # n * beta0 + Xi.sum() * beta1 + Xi.pow2.sum() * beta2 = Yi.sum()
    # Xi.sum() * beta0 + Xi.pow(2).sum() + Xi.pow(3).sum()* beta2 = Xi.sum * Yi.sum()
    # Xi.pow(2).sum() * beta0 + Xi.pow(3).sum() + Xi.pow(4).sum() * beta2 = Xi.pow(2).sum * Yi.sum()

    if (Xi.shape != Yi.shape):
        print("error Xi shape is not smae as Yi shape")
        return 0, 0

    Xi_sum      = Xi.sum()
    Xi_pow2_sum = np.power(Xi, 2).sum()
    Xi_pow3_sum = np.power(Xi, 3).sum()
    Xi_pow4_sum = np.power(Xi, 4).sum()
    
    Yi_sum      = Yi.sum()

    Xi_mul_Yi_sum      = (Xi * Yi).sum()
    Xi_pow2_mul_Yi_sum = (np.power(Xi, 2) * Yi).sum()

    n                  = Xi.size


    print("{}Beta0 + {}beta1 + {}beta2 = {}".format(n, Xi_sum, Xi_pow2_sum, Yi_sum))
    print("{}Beta0 + {}beta1 + {}beta2 = {}".format(Xi_sum, Xi_pow2_sum, Xi_pow3_sum, Xi_mul_Yi_sum))
    print("{}Beta0 + {}beta1 + {}beta2 = {}".format(Xi_pow2_sum, Xi_pow3_sum, Xi_pow4_sum, Xi_pow2_mul_Yi_sum))

    # clac betas
    xx = np.array([
        [n, Xi_sum, Xi_pow2_sum],
        [Xi_sum, Xi_pow2_sum, Xi_pow3_sum],
        [Xi_pow2_sum, Xi_pow3_sum, Xi_pow4_sum]
    ])

    xy = np.array([
        [Yi_sum],
        [Xi_mul_Yi_sum],
        [Xi_pow2_mul_Yi_sum]
    ])

    betas = np.matmul(np.linalg.inv(xx), xy)

    print("#############################")
    print("betas = (X'X)**-1 . X'Y = ")
    print(np.linalg.inv(xx))
    print(".")
    print(xy)
    print("=")
    print(betas)
    print("Beta0 is {}, Beta1 is {}, Beta2 is {}".format(betas[0][0], betas[1][0], betas[2][0]))
    print("#############################")

    return betas[0][0], betas[1][0], betas[2][0]

## Provede předpověď Y pro X
# @param beta0
# @param beta1
# @param beta2
# @param X
# @retrn Predicted value
def regreseHyperbolaPredict(beta0, beta1, beta2, x):
    y = beta0 + (beta1 * x) + (beta2 * x ** 2)
    print("#############################")
    print("Y = beta0 + beta1 * x + beta2 * x ** 2 = {}".format(y))
    print("#############################")
    return y

# get Se for regreseHyperbola
# @param beta0
# @param beta1
# @param beta2
# @param Xi
# @param Yi
# @retrn Se
def regreseHyperbolaGetSe(beta0, beta1, beta2, Xi, Yi):
    Yi_pow2_sum = np.power(Yi, 2).sum()
    Yi_sum      = Yi.sum()

    Xi_mul_Yi_sum = (Xi * Yi).sum()
    Xi_pow2_mul_Yi_sum = (np.power(Xi, 2) * Yi).sum()

    Se = Yi_pow2_sum - beta0 * Yi_sum - beta1 * Xi_mul_Yi_sum - beta2 * Xi_pow2_mul_Yi_sum

    print("#####################")
    print("Se = {} - {} * {} - {} * {} - {} * {} = {}".format(
         Yi_pow2_sum,
         beta0,
         Yi_sum,
         beta1,
         Xi_mul_Yi_sum,
         beta2,
         Xi_pow2_mul_Yi_sum,
         Se
    ))
    print("#####################")

    return Se

# Get SigmaPow2 regreseHyperbola
# @param beta0
# @param beta1
# @param beta3
# @param Xi
# @param Yi
# @retrn Sigma**2
def regreseHyperbolaGetSigmaPow2(beta0, beta1, beta2, Xi, Yi):
    Se = regreseHyperbolaGetSe(beta0, beta1, beta2, Xi, Yi)
    n  = Xi.size

    SigmaPow2 = Se / (n-3)

    print("#####################")
    print("SigmaPow2 = s**2 = Se / (n-3) = {} / ({} - 3) = {}".format(
         Se,
         n,
         SigmaPow2
    ))
    print("#####################")

    return SigmaPow2

# Get Qvality or regreseHyperbola in %
# @param beta0
# @param beta1
# @param beta2
# @param Xi
# @param Yi
# @retrn R**2
def regreseHyperolaGetQvality(beta0, beta1, beta2, Xi, Yi):
    prum = np.mean(Yi)

    Sypow2 = (1 / (Yi.size - 1)) * np.power(Yi - prum, 2).sum()

    print("#####################")
    print("Sy**2 = (1 / ( n - 1)) * suma((Yi - prum(Y)) ** 2) = {}".format(
         Sypow2
    ))
    print("#####################")

    St = Sypow2 * (Yi.size - 1)

    print("#####################")
    print("St = ( n - 1) * Sy**2 = {}".format(
         St
    ))
    print("#####################")

    Se = regreseHyperbolaGetSe(beta0, beta1, beta2, Xi, Yi)


    Rpow2 = 1 - Se / St

    print("#####################")
    print("R**2 = 1 - Se / St = 1 - {} / {} = {}".format(
         Se,
         St,
         Rpow2
    ))
    print("#####################")

    return Rpow2


###
# PLANE
##

## Pomocí regresní roviny popište závislost Yi na Xi a Zi
# @param Xi np.array of values
# @param Yi np.array of values
# @param Zi np.array of values
# @return beta0, beta1 and beta2
def regresePlaneBetas(Xi, Yi, Zi):
    # n * beta0 + Xi.sum() * beta1 + Zi.sum() * beta2 = Yi.sum()
    # Xi.sum() * beta0 + Xi.pow(2).sum() * beta1 + (Zi * Xi).sum() * beta2 = (Xi * Yi).sum()
    # Zi.sum() * beta0 + (Zi * Xi).sum() * beta1 + Zi.pow(2).sum() * beta2 = (Zi * Yi).sum()

    if ((Xi.shape != Yi.shape) and (Xi.shape != Zi.shape)):
        print("error Xi shape is not same as Yi shape or Zi shape")
        return 0, 0

    Xi_sum      = Xi.sum()
    Xi_pow2_sum = np.power(Xi, 2).sum()
    
    Zi_sum = Zi.sum()
    Zi_pow2_sum = np.power(Zi, 2).sum()

    Xi_mul_Zi_sum = (Zi * Xi).sum()


    
    Yi_sum        = Yi.sum()

    Xi_mul_Yi_sum = (Xi * Yi).sum()
    Zi_mul_Yi_sum = (Zi * Yi).sum()

    n             = Xi.size


    print("{}Beta0 + {}beta1 + {}beta2 = {}".format(n, Xi_sum, Zi_sum, Yi_sum))
    print("{}Beta0 + {}beta1 + {}beta2 = {}".format(Xi_sum, Xi_pow2_sum, Xi_mul_Zi_sum, Xi_mul_Yi_sum))
    print("{}Beta0 + {}beta1 + {}beta2 = {}".format(Zi_sum, Xi_mul_Zi_sum, Zi_pow2_sum, Zi_mul_Yi_sum))

    # clac betas
    xx = np.array([
        [n, Xi_sum, Zi_sum],
        [Xi_sum, Xi_pow2_sum, Xi_mul_Zi_sum],
        [Zi_sum, Xi_mul_Zi_sum, Zi_pow2_sum]
    ])

    xy = np.array([
        [Yi_sum],
        [Xi_mul_Yi_sum],
        [Zi_mul_Yi_sum]
    ])

    betas = np.matmul(np.linalg.inv(xx), xy)

    print("#############################")
    print("betas = (X'X)**-1 . X'Y = ")
    print(np.linalg.inv(xx))
    print(".")
    print(xy)
    print("=")
    print(betas)
    print("Beta0 is {}, Beta1 is {}, Beta2 is {}".format(betas[0][0], betas[1][0], betas[2][0]))
    print("#############################")

    return betas[0][0], betas[1][0], betas[2][0]


## do predict Y from X and Z 
# @param Xi np.array of values
# @param Yi np.array of values
# @param Zi np.array of values
# @return Y value
def regresePlanePredict(beta0, beta1, beta2, x, z):
    y = beta0 + (beta1 * x) + (beta2 * z)
    print("#############################")
    print("Y = beta0 + beta1 * x + beta2 * z = {}".format(y))
    print("#############################")
    return y 

# get Se for regresePlane
# @param beta0
# @param beta1
# @param beta2
# @param Xi
# @param Yi
# @param Zi
# @retrn Se
def regresePlaneGetSe(beta0, beta1, beta2, Xi, Yi, Zi):
    Yi_pow2_sum = np.power(Yi, 2).sum()
    Yi_sum      = Yi.sum()

    Xi_mul_Yi_sum = (Xi * Yi).sum()
    Zi_mul_Yi_sum = (Zi * Yi).sum()

    Se = Yi_pow2_sum - beta0 * Yi_sum - beta1 * Xi_mul_Yi_sum - beta2 * Zi_mul_Yi_sum

    print("#####################")
    print("Se = {} - {} * {} - {} * {} - {} * {} = {}".format(
         Yi_pow2_sum,
         beta0,
         Yi_sum,
         beta1,
         Xi_mul_Yi_sum,
         beta2,
         Zi_mul_Yi_sum,
         Se
    ))
    print("#####################")

    return Se

# Get SigmaPow2 regresePlane
# @param beta0
# @param beta1
# @param beta3
# @param Xi
# @param Yi
# @retrn Sigma**2
def regresePlaneGetSigmaPow2(beta0, beta1, beta2, Xi, Yi, Zi):
    Se = regresePlaneGetSe(beta0, beta1, beta2, Xi, Yi, Zi)
    n  = Xi.size

    SigmaPow2 = Se / (n-3)

    print("#####################")
    print("SigmaPow2 = s**2 = Se / (n-3) = {} / ({} - 3) = {}".format(
         Se,
         n,
         SigmaPow2
    ))
    print("#####################")

    return SigmaPow2

# Get Qvality or regreseHyperbola in %
# @param beta0
# @param beta1
# @param beta2
# @param Xi
# @param Yi
# @param Zi
# @retrn R**2
def regresePlaneGetQvality(beta0, beta1, beta2, Xi, Yi, Zi):
    prum = np.mean(Yi)

    Sypow2 = (1 / (Yi.size - 1)) * np.power(Yi - prum, 2).sum()

    print("#####################")
    print("Sy**2 = (1 / ( n - 1)) * suma((Yi - prum(Y)) ** 2) = {}".format(
         Sypow2
    ))
    print("#####################")

    St = Sypow2 * (Yi.size - 1)

    print("#####################")
    print("St = ( n - 1) * Sy**2 = {}".format(
         St
    ))
    print("#####################")

    Se = regresePlaneGetSe(beta0, beta1, beta2, Xi, Yi, Zi)


    Rpow2 = 1 - Se / St

    print("#####################")
    print("R**2 = 1 - Se / St = 1 - {} / {} = {}".format(
         Se,
         St,
         Rpow2
    ))
    print("#####################")

    return Rpow2

def PearsonKorelKoef(Xi, Yi):
    PrumX  = np.mean(Xi)
    PrumY  = np.mean(Yi)
    n      = Xi.size

    # Vzorec https://portal.matematickabiologie.cz/index.php?pg=aplikovana-analyza-klinickych-a-biologickych-dat--biostatistika-pro-matematickou-biologii--zaklady-korelacni-analyzy--pearsonuv-korelacni-koeficient--vypocet-pearsonova-korelacniho-koeficientu
    #p = ((Xi - PrumX) * (Yi - PrumY)).sum() / math.sqrt(
    #    np.power(Xi - PrumX, 2).sum() * np.power(Yi - PrumY, 2).sum()
    #)

    p = ((Xi * Yi).sum() - n * PrumX * PrumY) / np.sqrt(
        (np.power(Xi, 2).sum() - n * np.power(PrumX, 2)) *
        (np.power(Yi, 2).sum() - n * np.power(PrumY, 2))
    )

    return p

##
# Pouzivame kdyz je z normalniho rozdeleni a chceme dokazat neexistenci 
# linearni zavislosti
#
def XiYidependenceNormal(Xi, Yi):
    p = PearsonKorelKoef(Xi, Yi)
    print("#####################")
    print("p = {}".format(
         p
    ))

    n = Xi.size

    print("Pearsonův korelační koeficient")
    print("#####################")

    T = (p / (math.sqrt(1 - np.power(p, 2)))) * math.sqrt(n - 2)

    print("#####################")
    print("T = (p / sqrt(1 - p **2)) * sqrt(n - 2) = {}".format(
        T
    ))
    print("n = {}".format(
        n
    ))
    print("#####################")

    print("#####################")
    print("Dopočítej!!!!")
    print("H0: Xi a Yi jsou nezávislé")
    print("H1: Xi a Yi nejsou nezávislé")
    print("W = { T: |T| >= t(1 - apha / 2)(n-2)} <= zamítne H0 pokud do intervalu T padne")
    print("#####################")

def SpearmanKorelKoef(Xi, Yi):
    #je nutné seřadit podle velikosti
    XiSort = np.sort(Xi)
    YiSort = np.sort(Yi)

    #ted je nutné indexi oproti původnímu
    ri = np.array([])
    qi = np.array([])
    for x in Xi:
        indexes = np.where(XiSort == x)
        index   = np.mean(indexes)
        ri      = np.append(ri, np.array(index + 1))

    for y in Yi:
        indexes = np.where(YiSort == y)
        index   = np.mean(indexes)
        qi      = np.append(qi, np.array(index + 1))

    print("#####################")
    print("ri = {}".format(
        ri
    ))
    print("qi = {}".format(
        qi
    ))
    print("#####################")

    n  = Xi.size
    ps = 1 - (6 / (n * (n ** 2 - 1))) * np.power(ri - qi, 2).sum()

    print("#####################")
    print("n = {}".format(
        n
    ))
    print("ps = 1 - 6 / (n(n**2 - 1)) * suma((ri - qi)**2) = {}".format(
        ps
    ))
    print("#####################")

    return ps



##
## Non-correct
##
def XiYidependenceSpojit(Xi, Yi):
    ps = SpearmanKorelKoef(Xi, Yi)

    n = Xi.size
    T = (ps / math.sqrt(1 - (ps ** 2))) * math.sqrt(n - 2)

    print("#####################")
    print("T = ps / sqrt(1 - (ps ** 2)) * sqrt(n - 2) = {}".format(
        T
    ))
    print("#####################")
    
    print("#####################")
    print("Dopočítej!!!!")
    print("H0: Xi a Yi jsou nezávislé")
    print("H1: Xi a Yi nejsou nezávislé")
    print("W = { T: |T| >= t(1 - apha / 2)(n-2)} <= zamítne H0 (Pokud bude plati jsou závislé)")
    print("#####################")


##
# Linear
##
Xi = np.array([
    4, 4, 3, 4, 4, 6, 4, 7, 4, 4, 5, 4
])

Yi = np.array([
    311, 313, 237, 313, 281, 446, 314, 494, 322, 302, 387, 334
])

"""
beta0, beta1 = regreseLinearBetas(Xi, Yi)

Y = regreseLinearPredict(beta0, beta1, 2)
print("Y is {}".format(Y))

Se = regreseLinearGetSigmaPow2(beta0, beta1, Xi, Yi)
SigmaPow2 = regreseLinearGetSigmaPow2(beta0, beta1, Xi, Yi)

Rpow2 = regreseLinearGetQvality(beta0, beta1, Xi, Yi)
"""

##
# Hyperbola
#  dtype=np.float64
##
Xi = np.array([
    138, 140, 144, 146, 148, 152, 153, 157
])

Yi = np.array([
    5.3, 5.5, 5.8, 5.6, 5.1, 4.4, 4.1, 3.9
])

#beta0, beta1, beta2 = regreseHyperbolaBetas(Xi, Yi)
#regreseHyperbolaPredict(beta0, beta1, beta2, 150)
#regreseHyperbolaGetSigmaPow2(beta0, beta1, beta2, Xi, Yi)
#regreseHyperolaGetQvality(beta0, beta1, beta2, Xi, Yi)

##
# Rovina
##
Xi = np.array([
    25, 26, 25, 27, 28, 27
])

Yi = np.array([
    1.34, 1.16, 1.25, 1.32, 1.45, 1.37
]) # (Predikované)!!!!!!

Zi = np.array([
    7.5, 4.3, 6.4, 5.2, 7.9, 6.8
])

#beta0, beta1, beta2 = regresePlaneBetas(Xi, Yi, Zi)
#regresePlanePredict(beta0, beta1, beta2, 26, 7)
#regresePlaneGetSigmaPow2(beta0, beta1, beta2, Xi, Yi, Zi)
#regresePlaneGetQvality(beta0, beta1, beta2, Xi, Yi, Zi)

##
# Korelační analíza
##

Xi = np.array([
    47.8, 34.9, 44.8, 62.3, 36.2, 64.3, 49.1, 64.4, 43.2, 47, 56.4, 68.1, 50.9
])

Yi = np.array([
    9.2, 10.2, 11.9, 8, 15.3, 11.7, 16.6, 7.9, 7.2, 10.2, 9.4, 10.6, 12.8
])

# pro normální rozdělení
XiYidependenceNormal(Xi, Yi)

# pro spojité rozdělení
#XiYidependenceSpojit(Xi, Yi)
