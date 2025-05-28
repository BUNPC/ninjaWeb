import sys
import numpy as np


# -----------------------------------------------------
def numdigits(x, base=None):
    if not base:
        base=10

    # if not isinteger(x):
    #     return -1
    if not np.issubdtype(np.array(x).dtype, np.integer):
        return -1

    x = abs(x)

    if x==0:
        return 1

    if base == 10:
        if int(np.ceil(np.log10(x)))==np.log10(x):
            return int(np.log10(x)+1)
        return int(np.ceil(np.log10(x)))

    if base == 2:
        if int(np.ceil(np.log(x)))==np.log(x):
            return int(np.log(x)+1)
        return int(np.ceil(np.log(x)))



# -----------------------------------------------------
def isinteger(x):
    b = False
    if type(x) == int:
        b = True
    if type(x) == np.int:
        b = True
    if type(x) == np.int8:
        b = True
    if type(x) == np.int16:
        b = True
    if type(x) == np.int32:
        b = True
    if type(x) == np.uint8:
        b = True
    if type(x) == np.uint16:
        b = True
    if type(x) == np.uint32:
        b = True
    return b


# -----------------------------------------------------
def isBitSet(x, bidx):
    return (x & (1 << bidx)) > 0



# -----------------------------------------------------
def nanmean(a0, dim=None):
    a = a0.copy()
    if dim is None:
        k = np.where(np.isnan(a.flatten()))
        a[np.isnan(a)] = 0
        return np.sum(a.flatten())/(len(a.flatten())-len(k[0]))

    elif dim == 0:
        if a.ndim==3:
            n = np.array([[a.shape[0]] * a.shape[2]] * a.shape[1])
            m = np.mean(a, axis=0)
            k = np.where(np.isnan(m))
            for iK in range(0,len(k[0])):
                n[k[0][iK],k[1][iK]] = n[k[0][iK],k[1][iK]] - len(np.where(np.isnan( a[:,k[0][iK],k[1][iK]] ))[0])
        elif a.ndim==2:
            n = np.array([a.shape[0]] * a.shape[1])
            m = np.mean(a, axis=0)
            k = np.where(np.isnan(m))[0]
            for iK in range(0,len(k)):
                n[k[iK]] = n[k[iK]] - len(np.where(np.isnan( a[:,k[iK]] ))[0])
        a[np.isnan(a)] = 0
        return np.sum(a, axis=0) / n

    elif dim == 1:
        if a.ndim==3:
            n = np.array([[a.shape[1]] * a.shape[2]] * a.shape[0])
            m = np.mean(a, axis=1)
            k = np.where(np.isnan(m))
            for iK in range(0,len(k[0])):
                n[k[0][iK],k[1][iK]] = n[k[0][iK],k[1][iK]] - len(np.where(np.isnan( a[k[0][iK],:,k[1][iK]] ))[0])
        elif a.ndim == 2:
            n = np.array([a.shape[1]] * a.shape[0])
            m = np.mean(a, axis=1)
            k = np.where(np.isnan(m))[0]
            for iK in range(0,len(k)):
                n[k[iK]] = n[k[iK]] - len(np.where(np.isnan( a[k[iK],:] ))[0])
        a[np.isnan(a)] = 0
        return np.sum(a, axis=1) / n

    elif dim == 2:
        n = np.array([[a.shape[2]] * a.shape[1]] * a.shape[0])
        m = np.mean(a, axis=2)
        k = np.where(np.isnan(m))
        for iK in range(0,len(k[0])):
            n[k[0][iK],k[1][iK]] = n[k[0][iK],k[1][iK]] - len(np.where(np.isnan( a[k[0][iK],k[1][iK],:] ))[0])
        a[np.isnan(a)] = 0
        return np.sum(a, axis=2) / n

    return np.array([])



# ------------------------------------------------------------
def circshift(a):
    b = a.copy()
    N = a.shape[2]
    ii = np.arange(0,N)
    kk = (ii+1) % N
    b[:,:,ii] = a[:,:,kk]
    return b


# ------------------------------------------------------------
def bitget(x, b):
    val = np.uint8([0] * len(b))
    for ii in range(0,len(b)):
        val[ii] = (x & pow(2,b[ii])) > 0
    return val


# ------------------------------------------------------------
def ismember(a,b,options=None):
    if options == 'rows':
        if np.isscalar(a):
            a = [a]
        r = [0] * len(a)
        for ii in range(0,len(a)):
            if np.all(b==a[ii]):
                r[ii] = 1
    else:
        k = np.where(b==a)
        r = len(k[0]) > 0
    return r


def ismember_old(a, b, options=None):
    """
    Mimics MATLAB's ismember function.

    Parameters:
        a : array-like
            Elements or rows to check for in b.
        b : array-like
            Array to check against.
        options : str, optional
            If 'rows', compare rows instead of individual elements.

    Returns:
        r : list or bool
            Boolean array indicating membership for each element (or row) in a.
    """
    a = np.array(a)
    b = np.array(b)

    if options == 'rows':
        # Ensure a and b are 2D arrays
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)

        r = [any(np.array_equal(row, br) for br in b) for row in a]
    else:
        r = np.isin(a, b)

    return r.tolist() if isinstance(r, np.ndarray) else r



# ------------------------------------------------------------
def testNanMean():
    a = np.array([[251.,  54.,    183.,  np.nan,    248.,    146.,     91.,    196.,  55.,  np.nan,  285.],
                  [257., 238.,    201.,    242.,    168.,     74.,     28.,  np.nan,  36.,    170.,  259.],
                  [255., 192.,     65.,    187.,    283.,  np.nan,    108.,    217.,  43.,    286.,  101.],
                  [196., 134.,    222.,    113.,     71.,     26.,    233.,     39.,  44.,  np.nan,  189.],
                  [280.,  84.,    298.,    275.,    245.,    167.,    263.,    163., 250.,    133.,  232.],
                  [299.,  22.,  np.nan,    163.,     67.,     17.,    108.,  np.nan, 143.,     13.,  116.],
                  [174., 297.,     52.,    228.,    195.,    100.,    163.,    132.,   9.,     75.,  299.],
                  [ 95., 189.,     94.,     10.,  np.nan,    232.,  np.nan,    122.,  46.,    193.,  277.]])

    b = np.array([[ 56., 454.,    183.,  np.nan,    232.,    112.,    122.,    106., 155.,    111.,  285.],
                  [157., 438.,    201.,    202.,    133.,     74.,    128.,    305., 236.,     79.,   59.],
                  [235., 392.,    165.,     87.,     83.,    552.,      8.,    207., 243.,     89.,  101.],
                  [496., 234.,    252.,  np.nan,    171.,    116.,    133.,    239., 144.,    288.,  189.],
                  [ 80., 184.,    298.,     75.,    145.,  np.nan,     24.,    103.,  50.,    443.,   32.],
                  [209.,   2.,     11.,  np.nan,    167.,     17.,    108.,    991., 103.,    313.,  216.],
                  [120., 297.,     52.,    228.,    300.,      0.,     33.,    232.,  29.,    205.,   99.],
                  [ 95., 189.,     94.,  np.nan,    453.,    232.,    234.,    172., 129.,    222.,  207.]])

    c = np.array([[651.,    89.,    553.,     45.,    248.,    146.,     91.,    196.,  55.,    270.,   285.],
                  [257.,   238.,    211.,    242.,    168.,     74.,     28.,  np.nan,  36.,    170.,   259.],
                  [ 55.,   124.,  np.nan,  np.nan,    283.,  np.nan,    108.,    217.,  43.,    286.,   101.],
                  [196.,    67.,    242.,    103.,     71.,     26.,    233.,     39.,  44.,    873.,   189.],
                  [200.,    84.,    208.,    205.,    245.,    167.,    263.,    163., 250.,    133.,   232.],
                  [111.,   122.,    334.,    103.,     67.,  np.nan,    108.,     90., 143.,     13.,   116.],
                  [124.,   207.,    552.,    228.,    195.,    100.,    163.,    132.,   9.,     75., np.nan],
                  [195., np.nan,    294.,    710.,    342.,    232.,    342.,    122.,  46.,    193.,   277.]])



    d = np.array([[[0.] * 11] * 8] * 3)
    d[0,:,:] = a
    d[1,:,:] = b
    d[2,:,:] = c

    m0 = np.round(nanmean(d))

    m1 = np.round(nanmean(d, 0))
    m2 = np.round(nanmean(d, 1))
    m3 = np.round(nanmean(d, 2))
    m4 = np.round(nanmean(m3, 0))

    sys.stdout.write('\n\n')

    sys.stdout.write('Results:  m0 = %0.f\n'% m0)

    sys.stdout.write('m1 = \n')
    print(np.round(m1))
    sys.stdout.write('\n')

    sys.stdout.write('m2 = \n')
    print(np.round(m2))
    sys.stdout.write('\n')

    sys.stdout.write('m3 = \n')
    print(np.round(m3))
    sys.stdout.write('\n')

    sys.stdout.write('m4 = \n')
    print(np.round(m4))
    sys.stdout.write('\n')

    sys.stdout.write('\n')



# ------------------------------------------------------------
def testCircShift():
    a = np.array([[[0.] * 6] * 3] * 4)
    a[:,:,0] = [[174,    22,   344], [75,   377,   180], [293,  121, 368], [131,   221,   197]]
    a[:,:,1] = [[342,   165,   411], [352,   212,  215], [221,  135,  444], [10,    99,   196]]
    a[:,:,2] = [[385,   189,   164], [198,   108,   336], [404,   395,   219], [378,   475,   417]]
    a[:,:,3] = [[384,   257,   100], [84,   442,  203], [431,   294,   374], [495,    77,   413]]
    a[:,:,4] = [[395,    56,    95], [159,   68,   248], [267,   339,    74], [45,   248,    27]]
    a[:,:,5] = [[425,   291,     0], [280,   408,   433], [465,   440,   306], [348,   494,   495]]
    return circshift(a)


# ------------------------------------------------------------
def testIsMember():
    a = np.uint8([23,48,6,22,98])
    b = ismember([3,8,48,0,22],a)
    print(b)


# ------------------------------------------------------------
if __name__ == "__main__":
    # testNanMean()
    # b = testCircShift()
    testIsMember()

