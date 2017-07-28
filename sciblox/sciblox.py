#------------- Daniel Han-Chen 2017
#------------- https://github.com/danielhanchen/sciblox
#------------- SciBlox v0.2.1
#-------------

maxcats = 15
import warnings
warnings.filterwarnings("ignore")
true = True; TRUE = True
false = False; FALSE = False

#-----------------------------
import pandas as pd, numpy as np, scipy, sklearn as sk, seaborn as sb
from copy import copy
import matplotlib.pyplot as plt
try:
    from jupyterthemes import jtplot;
    jtplot.style()
except: pass;
#-----------------------------
np.set_printoptions(suppress = True)
pd.set_option('display.max_rows', 10)
pd_colour = '#302f2f'

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DATAFRAME METHODS ------------------------------------#
#-------------------- Display options and pandas methods --------------------#
def maxrows(x = 10): pd.set_option('display.max_rows', x)
def maxcat(x = 15): maxcats = x
def tabcolour(x = '#302f2f'): pd_colour = x
#-----------------------------
def percent(x):
    if x <= 1: return x
    else: return x/100
#-----------------------------
def table(x): 
    try: return pd.DataFrame(x)
    except: return pd.DataFrame(list(x.items()))
def series(x): 
    try: return pd.Series(x)
    except:
        first = pd.Series(x[0])
        if len(first)!=len(x): return pd.Series(T(x)[0])
        else: return first
#-----------------------------
def istable(x): return (type(x) in [pd.DataFrame,pd.Series])*1
def isarray(x): return (type(x) in [np.array,np.ndarray,np.matrix])*1
#-----------------------------
def shape(x):
    try: return x.shape
    except: return len(x)
#-----------------------------    
def head(x, n = 5):
    if istable(x)==1: return x.head(n)
    else:
        if len(x) > n: return x[:n]
        else: return x
        
def tail(x, n = 5):
    if istable(x)==1: return x.tail(n)
    else:
        if len(x) > n: return x[-n:]
        else: return x
#-----------------------------        
def sample(x, n = 5, ordered = False):
    if n > len(x): g = len(x)
    else: g = n
        
    if istable(x)==1:
        if ordered == False: return x.sample(g)
        else: return x.iloc[[int(y*(len(x)/g)) for y in range(g)]]
    else:
        if ordered == False: return np.random.choice(x, g)
        else: return np.array(x)[[int(y*(len(x)/g)) for y in range(g)]]
#-----------------------------        
def columns(x):
    try: return x.columns.tolist()
    except: pass;
        
def index(x):
    try: return x.index.tolist()
    except: pass;
#-----------------------------        
def reset(x, index = True, column = False, string = False, drop = False):
    if index == True and column == False: 
        if drop == False: return x.reset_index()
        else: return x.reset_index()[columns(x)]
    else:
        y = copy(x)
        if type(x)==pd.Series: ss = 0
        else: ss = shape(x)[1]
        if string == True: y.columns = ["col"+str(y) for y in range(ss)]
        else: y.columns = [y for y in range(ss)]
        return y
#-----------------------------        
def hcat(*args):
    a = args[0]
    if type(a)==pd.Series: a = table(a)
    for b in args[1:]:
        if type(a)==list:
            if type(b)!=list: b = list(b)
            a = a + b
        elif isarray(a)==1:
            if isarray(b)==0: b = array(b)
            a = np.hstack((a,b))
        else:
            if type(b)!=pd.DataFrame: b = table(b)
            a = pd.concat([a,b],1)
    del b
    return a

def vcat(*args):
    a = args[0]
    if type(a)==pd.Series: a = table(a)
    elif type(a)==list: a = array(a)
    for b in args[1:]:
        if isarray(a)==1:
            if isarray(b)==0: b = array(b)
            a = np.vstack((a,b))
        else:
            if type(b)!=pd.DataFrame: b = table(b)
            a = pd.concat([a,b],0)
    del b
    return a
#-----------------------------
def dtypes(x):
    if type(x)==pd.Series:
        types = x.dtype
        if types==('O' or "string" or "unicode"): return 'obj'
        elif types==("int64" or "uint8" or "uint16" or "uint32" or "uint64" or "int8" or "int32" or "int16"): return 'int'
        elif types==('float64' or 'float16' or 'float32' or 'float128'): return 'float'
        elif types=='bool': return 'bool'
        else: return 'date'
    else:
        dfs = x.dtypes
        for f in (dfs.index.tolist()):
            dfs[f] = str(dfs[f])
            if "int" in dfs[f]: dfs[f] = 'int'
            elif "float" in dfs[f]: dfs[f] = "float"
            elif "bool" in dfs[f]: dfs[f] = "bool"
            elif "O" in dfs[f] or "obj" in dfs[f]: dfs[f] = "obj"
            elif "date" in dfs[f]: dfs[f] = "date"
            else: dfs[f] = "obj"
        return dfs
def dtype(x): return dtypes(x)

def contcol(x): 
    try: return ((dtypes(x)=="int")|(dtypes(x)=="float")).index[(dtypes(x)=="int")|(dtypes(x)=="float")].tolist()
    except: return np.nan
def conts(x):
    if type(x) == pd.Series:
        if dtype(x) in ["int","float"]: return x
        else: return np.nan
    else: return x[contcol(x)]

def objcol(x): 
    try: return (dtypes(x)=="obj").index[dtypes(x)=="obj"].tolist()
    except: return np.nan
def objects(x):
    if type(x) == pd.Series:
        if dtype(x) == "obj": return x
        else: return np.nan
    else: return x[objcol(x)]
def objs(x): return objects(x)
def notobj(x): return exc(x, objcol(x))

def catcol(x):
    if type(x) == pd.Series:
        if iscat(x) == True: return x
        else: return np.nan
    else: return (iscat(x).index[iscat(x)]).tolist()
def classcol(x): return cats(x)
def cats(x): return x[catcol(x)]
def classes(x): return x[catcol(x)]

def iscat(x, cat = maxcats):
    return ((dtypes(x)!='float')|(dtypes(x)!='int'))&(nunique(x)<=cat)
#-----------------------------
def nullcol(x): return (count(x)!=len(x)).index[count(x)!=len(x)].tolist()
def nacol(x): return nullcol(x)
def missingcol(x): return nullcol(x)

def notnull(x, row = 1, keep = None, col = 0):
    if row!=1: axis = 1
    elif col!=0: axis = 0
    else: axis = 0
    if keep is None: 
        try: return x.dropna(axis = axis)
        except: return x.dropna()
    else:
        if keep < 1: 
            if axis==1: keep = len(x)*keep
            else: keep = shape(x)[1]*keep
        return x.dropna(axis = axis, thresh = keep)
    
def isnull(x, row = 1, keep = None, col = 0):
    if row!=1 or col!=0: axis = 0
    else: axis = 1
    if keep is None: miss = missing(x, row = axis)!=0
    else:
        if axis == 1:
            if keep < 1: miss = missing(x, row = axis)<=shape(x)[1]*keep
            else: miss = missing(x, row = axis)<=keep
        else:
            if keep < 1: miss = missing(x, row = axis)<=len(x)*keep
            else: miss = missing(x, row = axis)<=keep
    try: return x.iloc[miss.index[miss]]
    except: return x[pd.isnull(x)==True]
    
def dropna(x, col = None):
    if col is None: return x.dropna()
    else:
        if type(col)!=list: col = list(col)
        return x.dropna(subset = col)
    
#-----------------------------
def diff(want, rem):
    w = copy(want)
    for j in w:
        if j in rem: w.remove(j)
    for j in rem:
        if j in w: w.remove(j)
    return w

def exc(x, l):
    if type(l) == str: l = [l]
    return x[diff(columns(x),l)]

def drop(x, l): return exc(x, l), x[l]
def pop(x, l): return exc(x, l), x[l]

def append(l, r):
    g = copy(l);
    if type(g)!= list: g = [g]
    if type(r) == list:
        for a in r: g.append(a)
    else: g.append(r)
    return g

#-------------
#-------------
#-------------
#-------------
#------------------------------------ OTHER ANALYTICAL METHODS ------------------------------------#
#-------------------- Uniques and counting and sorting --------------------#
def count(x):
    try: return x.count()
    except: return len(x)
    
def missing(x, row = 0, col = 1):
    if row!=0 or col!=1: x = x.T 
    try: return (pd.isnull(x)).sum()
    except: return (np.isnan(x)).sum()
#-----------------------------    
def unique(x, dropna = False):
    if dropna == True: x = notnull(x)
    if type(x) == pd.Series: return list(x.unique())
    elif type(x) == pd.DataFrame: return {col:list(x[col].unique()) for col in columns(x)}
    else:
        u = []
        for a in x:
            if dropna == True: 
                if a not in u and a!=np.nan: u.append(a)
            else: 
                if a not in u: u.append(a)
        del a
        return u
    
def nunique(x, dropna = False):
    if istable(x)==True: return x.nunique()
    else:
        u = []; n = 0
        for a in x:
            if dropna == True: 
                if a not in u and a!=np.nan: u.append(a); n += 1
            else: 
                if a not in u: u.append(a); n += 1
        del u,a
        return n
    
def cunique(x, dropna = False):
    if type(x) == pd.Series: return x.value_counts(dropna = dropna)
    elif type(x) == pd.DataFrame: return {col:x[col].value_counts() for col in columns(x)}
    else:
        u = {}
        for a in x:
            if dropna == True: 
                if a not in u and a!=np.nan: u[a]=1
                else: u[a]+=1
            else: 
                if a not in u: u[a]=1
                else: u[a]+=1
        del a
        return u
    
def punique(x, dropna = False): 
    return round(nunique(x, dropna = dropna)/(count(x)+missing(x)*(dropna==False)*1)*100,4)
#-----------------------------
def reverse(x):
    if type(x) == pd.Series and dtype(x) == 'bool': return x == False
    elif istable(x)==1: return x.iloc[::-1]
    elif type(x) == list: return x[::-1]
    elif type(x) == dict: return {i[1]:i[0] for i in x.items()}
#-----------------------------
def sort(x, by = None, asc = True, ascending = True, des = False, descending = False):
    if type(x) == list:
        if asc == ascending == True and des == descending == False: return sorted(x)
        else: return reverse(sorted(x))
    else:
        if type(x) == pd.Series:
            if asc == ascending == True and des == descending == False: return x.sort_values(ascending = True)
            else: return x.sort_values(ascending = False)
        else:
            if by is None: col = columns(x)
            else: col = by
            if asc == ascending == True and des == descending == False: return x.sort_values(ascending = True, by = col)
            else: return x.sort_values(ascending = False, by = col)
            
def fsort(x, by = None, keep = False, asc = True, ascending = True, des = False, descending = False):
    if type(x)==pd.Series: x = table(x); x = reset(x, column = True, string = True); by = columns(x)[0];
    if type(by)==list: by = by[0]
    if type(x) == list:
        from collections import Counter
        c = copy(x)
        if asc == ascending == True and des == descending == False: c.sort(key=Counter(sort(c, asc = True)).get, reverse = True); return c
        else: c.sort(key=Counter(sort(c, asc = False)).get, reverse = False); return c
    elif by is None: print("Please specify column to sort by: fsort(x, by = 'Name')")
    else:
        f = by; fg = reset(table(x[f].value_counts()))
        ff = f+"_Freq"; fg.columns = [f,ff]
        del ff
        try: fg[f+"_Length"] = fg[f].str.len()
        except: fg[f+"_Length"] = fg[f]
            
        df = x.merge(fg, how = "outer")
        if asc == ascending == True and des == descending == False: df = sort(df, [f+"_Freq",f+"_Length"], asc = True)
        else: df = sort(df, [f+"_Freq",f+"_Length"], asc = False)
        
        if keep == True: return df
        else: l = columns(df);  l.remove(f+"_Freq");  l.remove(f+"_Length")
        return df[l]

#-------------
#-------------
#-------------
#-------------
#------------------------------------ BASIC ANALYSIS METHODS ------------------------------------#
#-------------------- Ratios and detections --------------------#
def freqratio(x):
    counted = cunique(x)
    if type(x) == pd.Series:
        try: return counted[0]/counted[1]
        except: return 1
    else:
        empty = []
        for col in columns(x):
            try: empty.append(counted[col].iloc[0]/counted[col].iloc[1])
            except: empty.append(1)
        tab = table(empty); tab.index = columns(x);        return tab[0]

def isid(x):
    for col in columns(x):
        if (nunique(x[col]) == len(x)) or "id" in col.lower() or "index" in col.lower(): return col
        else: return ''
        
def pzero(x): return sum(x==0, axis = 0)/count(x)*100

#-------------
#-------------
#-------------
#-------------
#------------------------------------ MATHEMATICAL METHODS ------------------------------------#
#-------------------- Statistical methods --------------------#
def var(x, axis = 0, dof = 1):
    try: return x.var(axis = axis, ddof = dof)
    except: return np.nanvar(x, axis = axis, ddof = dof)
        
def std(x, axis = 0, dof = 1):
    try: return x.std(axis = axis, ddof = dof)
    except: return np.nanstd(x, axis = axis, ddof = dof)

#-------------    
def mean(x, axis = 0):
    try: return x.mean(axis = axis)
    except: return np.nanmean(x, axis = axis)
    
def median(x, axis = 0):
    try: return x.median(axis = axis)
    except: return np.nanmedian(x, axis = axis)
    
def mode(x, axis = 0):
    try: return series(x).mode()[0]
    except: return x.mode(axis = axis).iloc[0]
    
def rng(x, axis = 0):
    try: return conts(x).max(axis = axis) - conts(x).min(axis = axis)
    except: 
        try: return max(x)-min(x)
        except: return np.nan
#-------------
def percentile(x, p, axis = 0):
    if p > 1: p = p/100
    try: return x.quantile(p, axis = axis)
    except: return np.nanpercentile(x, p, axis = axis)

def iqr(x, axis = 0):
    return percentile(x, 0.75, axis = axis) - percentile(x, 0.25, axis = axis)
#-------------
def skewness(x, axis = 0):
    try: return x.skew(axis = axis)
    except: return scipy.stats.skew(x, axis = axis, nan_policy='omit')
def skew(x, axis = 0): return skewness(x, axis)
    
def kurtosis(x, axis = 0):
    try: return scipy.stats.kurtosis(x, axis = axis, nan_policy='omit')
    except: return x.kurt(axis = axis)
    
def kurt(x, axis = 0): return kurtosis(x, axis)
#-------------
def pnorm(p, mean = 0, var = 1):
    if p > 1: p = p/100
    return scipy.stats.norm.cdf(p, loc=mean, scale=var)

def qnorm(q, mean = 0, var = 1):
    if q > 1: q = q/100
    return scipy.stats.norm.ppf(q, loc=mean, scale=var)

def CI(q, data, method = "mean",U = True, L = True):
    if q > 1: q = q/100
    norms = qnorm(q+(1-q)/2)*(std(data) / sqrt(len(data)) )
    if method == "mean": u = mean(data) + norms; l = mean(data) - norms
    if U == L == True: return (l,u)
    elif U == True: return u
    else: return l

#-------------
#-------------
#-------------
#-------------
#------------------------------------ TYPES METHODS ------------------------------------#
#-------------------- Changing case --------------------#
def lower(x):
    j = copy(x)
    if type(x) == list:
        for k in range(len(j)):
            try: j[k] = j[k].lower()
            except: pass;
    return j

def upper(x):
    j = copy(x)
    if type(x) == list:
        for k in range(len(j)):
            try: j[k] = j[k].upper()
            except: pass;
    return j
#-------------------- Other types and conversions --------------------#
def int(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = x[y].astype("int64")
            except: 
                try: x[y] = np.floor(x[y])
                except: pass
        return x
    else: 
        try: return np.int64(df)
        except: 
            try: return np.floor(df)
            except: return df
            
def float(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = x[y].astype("float64")
            except: pass
        return x
    else: 
        try: return np.float64(df)
        except: return df
#-------------        
def max(x, axis = 0):
    if istable(x)==1: return conts(x).max()
    else:
        if shape(matrix(x))[0] == 1: return np.amax(x,axis=axis)
        else: return np.amax(x)
        
def min(x, axis = 0):
    if istable(x)==1: return conts(x).min()
    else:
        if shape(matrix(x))[0] == 1: return np.amin(x)
        else: return np.amin(x,axis=axis)
#------------- 
def argcheck(df, args):
    if len(args)==0: col = columns(df)
    elif type(args[0])!=list: col = list(args)
    else: col = args[0]
    return copy(df), col
#-------------
def abs(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.abs(x[y])
            except: pass
        return x
    else: 
        try: return np.abs(df)
        except: return df
#-------------        
def log(df, *args, shift = 0):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.log(x[y]+shift)
            except: pass;
        return x
    else: 
        try: return np.log(df+shift)
        except: return df
#-------------        
def exp(df, *args, shift = 0):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.exp(x[y])+shift
            except: pass;
        return x
    else: 
        try: return np.exp(df)+shift
        except: return df
#-------------        
def sin(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.sin(x[y])
            except: pass;
        return x
    else: 
        try: return np.sin(df)
        except: return df
#-------------
def cos(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.cos(x[y])
            except: pass;
        return x
    else: 
        try: return np.cos(df)
        except: return df
#-------------        
def cos(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.cos(x[y])
            except: pass;
        return x
    else: 
        try: return np.cos(df)
        except: return df
#-------------        
def sqrt(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.sqrt(x[y])
            except: pass;
        return x
    else: 
        try: return np.sqrt(df)
        except: return df
#-------------        
def floor(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.floor(x[y])
            except: pass;
        return x
    else: 
        try: return np.floor(df)
        except: return df
#-------------        
def ceiling(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            try: x[y] = np.ceil(x[y])
            except: pass;
        return x
    else: 
        try: return np.ceil(df)
        except: return df

def ceil(df, *args): return ceiling(df, args)
#-------------
def sum(x, axis = 1):  
    try: return x.sum(axis = axis)
    except: return np.nansum(x, axis = 0)

#-------------
#-------------
#-------------
#-------------
#------------------------------------ MATHEMATICAL METHODS ------------------------------------#
#-------------------- Linear Algebra --------------------#
from numpy import dot, multiply, multiply as mult
def array(*args):
    if len(args)==1: 
        arrs = np.array(args[0])
        try:
            if shape(arrs)[1]==1: arrs = arrs.T[0]
        except: pass;
        return arrs
    else:
        try: return np.array(args)
        except: return np.array([args])

def matrix(*args): return np.matrix(array(args))

def T(x):
    if type(x)==np.array: return matrix(x).T
    else: 
        try: return x.T
        except: return array(x).T
        
def inv(x):
    try: return np.linalg.inv(x)
    except: print("Either det(x)=0 or not square matrix")
        
def det(x):
    try: return np.linalg.det(x)
    except: print("Not square matrix")
#-------------         
def eye(x): return np.eye(x)
def I(x): return np.eye(x)
#------------- 
def ones(x, s = 1):
    if s == 1: return np.ones((x,x))
    else: return np.ones(x)
    
def J(x, s = 1): return ones(x, s)
#-------------     
def zeros(x, s = 1):
    if s == 1: return np.zeros((x,x))
    else: return np.zeros(x)
    
def zeroes(x, s = 1): return zeros(x, s)
def Z(x, s = 1): return zeros(x, s)
#------------- 
def triu(matrix): return np.triu(matrix)
def tril(matrix): return np.tril(matrix)
#------------- 
def trace(A): return np.trace(A)
def tr(A): return trace(A)
def diag(A): return np.diagonal(A)
#------------- 
def repmat(A, *args):
    if len(args) == 2: return np.tile(A, (args[0],args[1]))
    elif len(args) == 1: return np.tile(A, args[0])
    else: print("Error")

def tile(A, *args): return repmat(A, args)

#-------------
#-------------
#-------------
#-------------
#------------------------------------ TABLE METHODS ------------------------------------#
#-------------------- Opening and editing --------------------#
def read(x):
    if type(x) == list:
        for y in x: 
            if "csv" in y: return clean(pd.read_csv(y))
    else: 
        if "csv" in x: return clean(pd.read_csv(x))
#-------------        
def string(dfx, *args):
    x = copy(dfx); df = copy(dfx)
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            x[y] = x[y].astype("str")+"*"
        return x
    elif type(df) == pd.Series: 
        df = df.astype("str")+"*"
        return df
    else: return str(df)
#-------------    
def clean(x, *args):
    def cleancol(x):
        if dtypes(x) == 'obj': 
            c = x.str.replace(",","").str.replace(" ","").str.replace("-","").str.replace("%","").str.replace("#","")
        else: c = x
            
        try: 
            if ((sum(int(c)) - sum(float(c)) == 0) or sum(int(c)-float(c))==0) and count(c) == len(c): return int(c)
            else: return float(c)
        except: 
            return x
            
    x = x.replace(np.inf, np.nan).replace(-np.inf, np.nan).replace("NaN",np.nan)
    
    df = copy(x)
    if type(x) == pd.Series: return cleancol(x)
    else:
        if len(args)==0: col = columns(df)
        elif type(args[0]) != list: col = list(args)
        else: col = args[0]
        for y in col: df[y] = cleancol(df[y])
        return df

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DATA ANALYTICS ------------------------------------#
#-------------------- Analyse --------------------#
def analyse(c, y = None, extra = ["skew"], dec = 2, colour = True, limit = True, graph = True):
    x = copy(c)
    if y is not None:
        if type(y) == str: x, y = drop(x, y)
    first = describe(x, extra = extra, clean = False); cols = columns(first)
    df = hcat(guess_importance(x,y), first)
    df.columns = append("Importance", cols)
    df = round(sort(df, by = ["Importance","FreqRatio","%Unique"], des = True),dec)
    if limit == True: df = df[df["Importance"]>0]
    if graph == True: plot(x = index(df)[0], y = index(df)[1], z = index(df)[2], hue = y, data = c)
    if colour == True: df = df.style.bar(align='mid', color=pd_colour, width = 80).set_properties(**{'max-width': '90px'})
    return df

def describe(x, extra = ["skew"], clean = True):
    normal = hcat(mean(x), median(x), rng(x), freqratio(x), mode(x), punique(x))
    normal.columns = ["Mean","Median","Range", "FreqRatio", "Mode","%Unique"]
    if type(extra)!=list: extra = [extra];extra = lower(extra); 
    for j in extra:
        before = columns(normal)
        if "skew" in j: normal = hcat(normal, skew(x)); normal.columns = append(before, "Skewness")
        elif "cat" in j: normal = hcat(normal, iscat(x)); normal.columns = append(before, "IsCategorical")
        elif "iqr" in j: normal = hcat(normal, iqr(x)); normal.columns = append(before, "InterQuartileRng")
        elif "var" in j: normal = hcat(normal, var(x)); normal.columns = append(before, "Variance")
        elif "std" in j or "sd" in j: normal = hcat(normal, std(x)); normal.columns = append(before, "SD")
        elif "min" in j: normal = hcat(normal, np.min(x)); normal.columns = append(before, "Min")
        elif "kurt" in j: normal = hcat(normal, kurtosis(x)); normal.columns = append(before, "Kurt")
        elif "max" in j: normal = hcat(normal, np.max(x)); normal.columns = append(before, "Max")
        elif "punq" in j: normal = hcat(normal, punique(x)); normal.columns = append(before, "%Unique")
        elif "nunq" in j: normal = hcat(normal, nunique(x)); normal.columns = append(before, "No.Unique")
    df = sort(normal, by = "FreqRatio")
    if clean == True: return df.replace(np.nan,"")
    else: return df
#-------------------- Var-Check and FreqRatio Check --------------------#    
def varcheck(x, freq = "mean", unq = 0.1, colour = True, limit = True, output = False):
    freqs = freqratio(x); unqs = punique(x)
    if freq == "mean": fd = (freqs>=CI(q=0.99,data =freqs,L=False))*1
    else: fd = (freqs>freq)*1
    df = hcat(freqs,fd,unqs,(unqs<=unq)*1,var(x))
    df.columns = ["FreqRatio","BadFreq?","%Unique","BadUnq?","Var"]
    df["BadVar?"] = (df["Var"].fillna(1000)<=0.1)*1
    df["BAD?"] = (df["BadFreq?"]+df["BadUnq?"]+df["BadVar?"])>0
    df = round(sort(df, by =["BAD?","BadVar?","BadFreq?","BadUnq?","FreqRatio","%Unique","Var"], des = True),2)
    
    if limit == True: df = T(T(df)[((df["BAD?"]==True).index[df["BAD?"]==True]).tolist()])
    if colour == True: 
        df = df.style.bar(align='zero', color=pd_colour, width = 80, subset=["FreqRatio","%Unique","Var"])
        df = df.apply(highlight_one, subset = ["BadFreq?","BadUnq?","BadVar?"]).apply(highlight_true, subset=["BAD?"])
        df = df.set_properties(**{'max-width': '90px'})
    if output == True: return exc(x, index(df))
    else: return df
    
#-------------------- Correlations --------------------# 
def corr(x, table = False, limit = 20):
    if table == False:
        corrs = round(x.corr()*100)
        sortby = sort(sum(abs(corrs)-100),des=False)
        corrs = corrs[index(sortby)]
        corrs = T(T(corrs)[index(sortby)])
        if shape(corrs)[0]>limit: corrs = T(T(corrs.iloc[0:limit]).iloc[0:limit])
        corrs = T(reverse(T(reverse(corrs))))
        cmap = sb.light_palette("black", as_cmap=True)
        show = abs(corrs).style.background_gradient(cmap).set_properties(**{'max-width': '50px', 'font-size': '8pt'
                                                                              ,'color':'black'})
        return show
    else:
        try: return conts(x).corr()
        except: print("Error. No continuous data")
            
def correlation(x, table = False): return corr(x, table)
def correlation_matrix(x, table = False): return corr(x, table)
def cor(x, table = False): return corr(x, table)

#-------------------- Feature Importance --------------------#
def guess_importance(df, y):
    x = copy(df)
    if type(y) == str:
        try: y = x[y]
        except: 
            print("No column for y")
    x = dummies(x)
    x_train, x_test, y_train, y_test = holdout(x, y, info = False);
    
    def lightmodel(x_train, x_test, y_train, y_test, reg, seed = 1234):
        try: import lightgbm as lgb
        except: print("Cannot install"); raise
        x_train = array(x_train); y_train = array(y_train); x_test = array(x_test); y_test = array(y_test)
        if reg == True:
            model = lgb.LGBMRegressor(objective='regression', num_leaves = 5, learning_rate = 0.1, n_estimators = 100, seed = seed)
            model.fit(x_train, y_train, early_stopping_rounds = 10, eval_metric='l2', eval_set=[(x_test, y_test)],verbose=False)
            return model
    
    imps = lightmodel(x_train, x_test, y_train, y_test, reg = True).feature_importances_
    tab = table(imps); tab.index = columns(x)
    imps = dict(tab)[0]*100; cols = columns(df)

    imp = {k:0 for k in cols}
    for j in imps.keys():
        for c in cols:
            if c in j: imp[c] += imps[j]
    return series(imp)

def guess_imp(df, y): return guess_importance(df, y)

#-------------------- Data Reduction --------------------#
## https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
def remcor(x, limit = 0.9):
    dataset = copy(x)
    col_corr = set(); corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= limit:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns: del dataset[colname]
    return dataset
def remcorr(x,limit=0.9): return remcor(x,limit)
#-------------
#https://stackoverflow.com/questions/28816627/how-to-find-linearly-independent-rows-from-a-matrix
def independent(A):
    try: import sympy
    except: 
        print("Cannot install"); raise
    _, inds = sympy.Matrix(A).T.rref()
    print("Lin Indp rows are: "+str(inds))
    return A[list(inds)]

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DUMMIFICATION ------------------------------------#
#-------------------- Dummies method --------------------#
def dummies(x, dummies = True, codes = False, freq = True, na = "inplace", nanew = True, col = None, ascending = True, cat = True, drop = True,
           ids = False):
    try:
        if dtypes(x)[0]==('int' or 'float') and type(x)==pd.Series: return x
    except:
        if dtypes(x)==('int' or 'float') and type(x)==pd.Series: return x
    if type(x)!=pd.DataFrame: x = table(x)
    df = copy(x)
    if ids == False: df = exc(df, isid(df))
    if col is None: 
        if cat == True: col = catcol(df)
        else: col = objcol(df)
    elif type(col)!=list: col = [col]
    if dummies == True:
        if "in" in na:
            for j in col:
                dummified = pd.get_dummies(x[j], dummy_na = nanew)
                dummified.columns = [str(j)+"_"+str(c) for c in columns(dummified)]
                if j in nacol(x): dummified.iloc[isnull(x[j]).index]=np.nan
                df = hcat(df, dummified)
        else: df = pd.get_dummies(x, dummy_na = nanew, columns = col)
        if drop == True: return notobj(zerodrop(df))
        else: return zerodrop(df)
    else:
        if freq == True:
            code = {}
            for j in col:
                part = {}; 
                try: i = min(df[j]);
                except: i = 0;
                if dtype(df[j])!=('int'or'float'): d = fsort(df, by = j)[j]
                else: d = sort(df, by = j)[j]
                for k in d:
                    if pd.isnull(k)==False:
                        try: part[k]
                        except: part[k] = i; i+=1
                code[j] = part
                df[j]=df[j].replace(part)
                del part,i,d,k
        else:
            code = {}
            for j in col: 
                code[j] = reverse(dict(enumerate(df[j].astype("category").cat.categories)))
                df[j]=df[j].replace(code[j])
        if drop == True: df = notobj(df)
        if shape(df)[1]==1: df = df[columns(df)[0]]
        if codes == True: return df,code
        else: return df
#-------------------- Quantile conversion --------------------#
def discretise(x, n = 4, smooth = True, codes = False):
    if codes == False: codes = None
    else: codes = False
    if smooth == True:
        try: return pd.qcut(x, q = n, duplicates = 'drop', labels = codes)
        except: return pd.cut(x, q = n, labels = codes)
    else:
        return pd.cut(x, bins = n, labels = codes)
        
def qcut(x, n = 4, smooth = True, codes = False): return discretise(x, n, smooth, codes)

#-------------
#-------------
#-------------
#-------------
#------------------------------------ ADVANCED DATA ANALYTICS ------------------------------------#
#-------------------- Skewness Analysis --------------------#
def topositive(y, info = False):
    x = copy(y); d = conts(x)
    notgood = ((np.min(d)<=0).index[np.min(d)<=0]).tolist()
    add = np.abs(np.min(d[notgood]))+1
    d[notgood] = d[notgood]+add
    x[columns(d)] = d
    if info == False: return x
    else: return x,add
#------------- 
def boxcox(x):
    if type(x) == pd.Series: 
        k = (conts(x)+abs(min(conts(x)))+1)
        lm = scipy.stats.boxcox(k)[1]
        if lm == 0: return log(x), lm
        else: return ((k**lm)-1)/lm, lm
    else:
        df = []; lms = []
        for col in contcol(x):
            k = (x[col]+abs(min(x[col]))+1)
            lm = scipy.stats.boxcox(k)[1]
            if lm == 0: df.append(log(x[col])); lms.append(lm)
            else: df.append(((k**lm)-1)/lm); lms.append(lm)
        return T(table(df)), array(lms)
#------------- 
def unskew(x, info = False):
    def show(q, df):
        if q == 0: return (df, "normal")
        elif q == 1: return (sqrt(df), "sqrt")
        else: return (boxcox(df)[0], "boxcox")
    original = copy(x)
    df = topositive(conts(x))
    skews = np.abs(skew(df))
    sqrted = sqrt(df)
    boxcoxed = boxcox(df)[0]
    comp = hcat(skew(df),skew(sqrted),skew(boxcoxed));    comp.columns = ["norm","sqrt","box"]
    res = np.abs(comp.T)
    r = []; out = []
    for col in res:
        p = 0
        for i in res[col]:
            if i == np.min(res[col]):
                f = show(p, df[col]); r.append(f[1]); out.append(f[0]); break
            else: p += 1
    first = out[0]
    for c in out[1:]: first = hcat(first, c)
        
    del c, out, res, comp, sqrted, skews, boxcoxed, show
    original[columns(first)] = first
    res = table(r); res.index = columns(first)
    
    if info == True: return original, res[0]
    else: return original
#-------------     
def outlier(df, method = "forest", poutlier = 0.025, sd = 3.5, iqr = 1.5, indicate = True, n_estimators = 100):
    x = copy(df)
    if "for" in method or "tree" in method:
        from sklearn.ensemble import IsolationForest
        df = dummies(x, na = "clear");       df = df.fillna(df[nullcol].median())
        model = IsolationForest(n_estimators = n_estimators, n_jobs=-1, bootstrap = True, contamination = poutlier)
        model.fit(df);            preds = model.predict(df)
        res = x.iloc[np.where(preds==-1)[0]]
    else:
        f = dummies(x, na = "clear");            df = topositive(f.fillna(f.median()))
        if "std" in method or "sd" in method:
            #https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
            if len(shape(df)) == 1: df = df[:,None]
            df = unskew(df)
            meds = median(df, axis=0)
            diff = sum((df - meds)**2, axis=1)
            diff = sqrt(diff);            mad = median(diff)
            z = 0.6745 * diff / mad
            out = (z>sd)==True
            where = out.index[out].tolist()
            res = x.iloc[where]

        elif "iqr" in method:
            first = percentile(df, p = 0.25)
            last = percentile(df, p = 0.75)
            iqrred = first-last
            where = sum((df>(last+iqr*last))|(df<(first-iqr*first)))!=0
            res = x.iloc[where.index[where].tolist()]
        
    print("No. outliers = "+str(len(res)))
    if indicate == True:
        x["IsOutlier"] = 0
        try: x["IsOutlier"].iloc[[res.index.tolist()]] = 1
        except: pass;
        return x
    else: return res
    
def isoutlier(df, method = "forest", poutlier = 0.025, sd = 3.5, iqr = 1.5, indicate = False, n_estimators = 100):
    d = outlier(df, method = method, poutlier = poutlier, sd = sd, iqr = iqr, indicate = True, n_estimators = n_estimators)
    if indicate == False: return exc(d.iloc[(d["IsOutlier"]==1).index[d["IsOutlier"]==1]], "IsOutlier")
    else: return d.iloc[(d["IsOutlier"]==1).index[d["IsOutlier"]==1]]

def notoutlier(df, method = "forest", poutlier = 0.025, sd = 3.5, iqr = 1.5, indicate = False, n_estimators = 100):
    d = outlier(df, method = method, poutlier = poutlier, sd = sd, iqr = iqr, indicate = True, n_estimators = n_estimators)
    if indicate == False: return exc(d.iloc[(d["IsOutlier"]==0).index[d["IsOutlier"]==0]], "IsOutlier")
    else: return d.iloc[(d["IsOutlier"]==0).index[d["IsOutlier"]==0]]
#-------------
def zerodrop(x): return exc(x, (pzero(x)==100).index[pzero(x)==100].tolist())

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DATA CLEANING AND CONVERSION ------------------------------------#
#-------------------- Normal statistic filling --------------------#
def fillobj(x, method):
    data = copy(clean(x))
    missed = nacol(data[objcol(data)]); missdf = data[missed]
    if method in ["mode","freq","frequency"]: data[missed] = data[missed].fillna(mode(missdf))
    elif method in ["zero","missing","none"]: data[missed] = data[missed].fillna("Missing_Data")
    elif method in ["mix","half","halved"]:
        ins = (count(x)<0.75*len(x)).index[count(x)<0.75*len(x)]
        data[ins] = data[ins].fillna("Missing_Data")
        other = diff(columns(x), ins)
        data[other] = data[other].fillna(mode(x[other]))
    return data
#-------------
def fillcont(x, method):
    data = copy(clean(x))
    missed = nacol(conts(data)); missdf = data[missed]
    if method in ["mean","avg","average"]: data[missed] = data[missed].fillna(mean(missdf))
    elif method in ["median"]: data[missed] = data[missed].fillna(median(missdf))
    elif method in ["mode","freq","frequency"]: data[missed] = data[missed].fillna(mode(missdf))
    return data
#-------------------- Full methods --------------------#
def complete(df, method = None, objects = None, continuous = None, knn = 5, max_unique = 20, epoch = 100, mice = "forest", ids = False):
    x = copy(df);       imputation = ["bpca","pca","knn","mice","svd"];               imped = 0
    if ids == False: x = exc(x, isid(x))
    if method is not None: meth = method.lower()
    else: meth = "a"
    if method is None and objects is None and continuous is None: meth = 'knn'
        
    if meth in imputation or objects in imputation or continuous in imputation:
        imped = 1
        try: import fancyimpute
        except: 
            print("Cannot import"); raise
            
        def matching(method, objects, continuous, thingo):
            if method is not None:
                if thingo in method: return 1
                else: return 0
            else:
                if thingo in objects or thingo in continuous: return 1
                else: return 0
            
        res,codes = dummies(x, codes = True, dummies = False)
        intcols = (dtypes(res)=='int').index[dtypes(res)=='int'].tolist()

        if matching(meth, objects, continuous, "knn") == 1: dfilled = fancyimpute.KNN(k=knn, verbose = 0).complete(res)
        elif matching(meth, objects, continuous, "svd") == 1: dfilled = fancyimpute.SoftImpute(verbose = 0).complete(res)
        elif matching(meth, objects, continuous, "mice") == 1: 
            print("Please wait...")
            dfilled = mice_complete(res, epochs = int(epoch/10), impute_method = mice, strings = objcol(x))
            print("Done")
        else: 
            print("Please wait...")
            dfilled = bpca_complete(res, epochs = epoch)
            print("Done")

        dfilled = table(dfilled); dfilled.columns = columns(res)

        for col in codes: x[col] = squeeze(series(int(round(dfilled[col],0))), upper = len(codes[col])-1, lower = 0).replace(reverse(codes[col]))
        for col in contcol(x): x[col] = dfilled[col]
        for col in contcol(x): x[col] = squeeze(x[col], lower = np.min(df[col]), upper = np.max(df[col]))

    if (missingcol(x) != [] and objects in imputation) or meth in imputation: x = fillobj(x, "mix")
    elif objects is not None: x[objcol(x)] = fillobj(df[objcol(df)], objects)

    if continuous not in imputation and continuous is not None: x[contcol(x)] = fillcont(df[contcol(df)], continuous)

    x = round(x, 4)
    x[intcols] = int(round(x[intcols]))
    return x

#-------------------- BPCA --------------------#
#http://ishiilab.jp/member/oba/tools/BPCAFill.html
def bpca_complete(x, epochs = 100):
    decimals = 4
    y = copy(x); cols = y.columns.tolist()
    maximum = np.int(np.max(y.max())*999)
    means = round(y.mean(),decimals); sd = round(y.std(),decimals); y = round((y-means)/sd,decimals)
    y[missingcol(y)] = y[missingcol(y)].fillna(maximum)
    mat = float(np.matrix(y))

    N,d = mat.shape; q = d-1
    yest = np.copy(mat); yest[yest==maximum]=0

    missidx = {};       bad = np.where(mat==maximum)
    for a in bad[0]: missidx[a] = []
    for a in range(len(bad[0])): missidx[bad[0][a]].append(bad[1][a])

    nomissidx = {};     good = np.where(mat!=maximum)
    for a in good[0]: nomissidx[a] = []
    for a in range(len(good[0])): nomissidx[good[0][a]].append(good[1][a])

    gmiss = list(set(bad[0]))
    gnomiss = list(set(good[0]))

    covy = np.cov(yest.T)
    U, S, V = np.linalg.svd(np.matrix(covy))
    U = (U.T[0:q]).T;         S = S[0:q]*np.eye(q);           V = (V.T[0:q]).T

    mu = np.copy(mat);        mu[mu==maximum]=np.nan;        mu = np.nanmean(mu, 0)
    W = U*np.sqrt(S);         tau = 1/ (np.trace(covy)-np.trace(S));      taumax = 1e20; taumin = 1e-20;    tau = np.amax([np.amin([tau,taumax]),taumin])

    galpha0 = 1e-10;          balpha0 = 1;                 alpha = (2*galpha0 + d)/(tau*np.diag(W.T*W)+2*galpha0/balpha0)
    gmu0  = 0.001;            btau0 = 1;                   gtau0 = 1e-10;                   SigW = eye(q)
    tauold = 1000

    for epoch in range(epochs):
        Rx = np.eye(q)+tau*W.T*W+SigW;            Rxinv = np.linalg.inv(Rx)
        idx = gnomiss; n = len(idx)                  
        dy = mat[idx,:] - np.tile(mu,(n,1));      x = tau * Rxinv * W.T * dy.T

        Td = dy.T*x.T;                            trS = np.sum(np.multiply(dy,dy))
        for n in range(len(gmiss)):
            i = gmiss[n]
            dyo = np.copy(mat)[i,nomissidx[i]] - mu[nomissidx[i]]
            Wm = W[missidx[i],:];                                  Wo = W[nomissidx[i],:]
            Rxinv = np.linalg.inv( Rx - tau*Wm.T*Wm );             ex = tau * Wo.T * np.matrix(dyo).T;   x = Rxinv * ex
            dym = Wm * x;                                          dy = np.copy(mat)[i,:]
            dy[nomissidx[i]] = dyo;                                dy[missidx[i]] = dym.T
            yest[i,:] = dy + mu
            Td = Td + np.matrix(dy).T*x.T;                            Td[missidx[i],:] = Td[missidx[i],:] + Wm * Rxinv
            trS = trS + dy*np.matrix(dy).T +  len(missidx[i])/tau + np.trace( Wm * Rxinv * Wm.T )

        Td = Td/N;                trS = trS/N;                        Rxinv = np.linalg.inv(Rx); 
        Dw = Rxinv + tau*Td.T*W*Rxinv + np.diag(alpha)/N;             Dwinv = np.linalg.inv(Dw);
        W = Td * Dwinv;

        tau = (d+2*gtau0/N)/(trS-np.trace(Td.T*W)  + (mu*np.matrix(mu).T*gmu0+2*gtau0/btau0)/N)[0,0];
        SigW = Dwinv*(d/N);
        alpha = (2*galpha0 + d)/ (tau*np.diag(W.T*W)+np.diag(SigW)+2*galpha0/balpha0).T

        if np.abs(np.log10(tau)-np.log10(tauold)) < 1e-4:  break;
        tauold = tau
    out = table(yest)
    out.columns = cols
    out = (out*sd)+means
    return out
#-------------------- MICE --------------------#
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
def mice_complete(res, strings, epochs = 10, impute_method = "forest"):
    x = copy(clean(res));     original = copy(x)
    filled = fillcont(original, method = "median")
    from sklearn.cross_validation import train_test_split
    
    for epoch in range(epochs):
        for missing_col in missingcol(original):
            null_data = isnull(original[missing_col]).index

            not_null = filled.iloc[notnull(original[missing_col]).index]
            y = not_null.pop(missing_col)
            
            if "forest" in impute_method or "tree" in impute_method or "bag" in impute_method:
                from sklearn.ensemble import RandomForestRegressor as rfr
                from sklearn.ensemble import RandomForestClassifier as rfc
                if missing_col in strings: model = rfc(n_jobs = -1, n_estimators=epochs*4)
                else: model = rfr(n_jobs = -1, n_estimators=epochs*4)
                    
            elif "linear" in impute_method or "log" in impute_method:
                from sklearn.linear_model import LinearRegression as linreg
                from sklearn.linear_model import LogisticRegression as logreg
                if missing_col in strings: model = logreg(n_jobs = -1, solver = 'sag', multi_class = "multinomial")
                else: model = linreg(n_jobs = -1)
                    
            elif "boost" in impute_method:
                from lightgbm import LGBMRegressor as xgbr
                from lightgbm import LGBMClassifier as xgbc
                if missing_col in strings: model = xgbc(learning_rate = 10/epochs, n_estimators=epochs*4, nthread =-1)
                else: model = xgbr(learning_rate = 10/epochs, n_estimators=epochs*4, nthread=-1)
                    
            train_x, test_x, train_y, test_y = train_test_split(not_null, y, test_size=0.33, random_state=42)
            model.fit(train_x, train_y)

            filled[missing_col].iloc[null_data] = model.predict(exc(filled.iloc[null_data], missing_col))
    return filled
#-------------------- Squeeze or round functions --------------------#
def squeeze(df, lower = 0, upper = 1):
    x = copy(df)
    x[x<lower] = lower; x[x>upper] = upper
    return x

#-------------
#-------------
#-------------
#-------------
#------------------------------------ MACHINE LEARNING ------------------------------------#
#-------------------- Boosting --------------------#
def lightgbm(x_train, x_test, y_train, y_test, noclass = None, lr = 0.05, method = "dart", gpu = False, trees = 100, metric = None, 
             depth = -1, splits=2, leaves=31.123, min_weight=20.123, features=1, bins=5.123, impurity=1e-3+0.000001, jobs=-1, state=None, bagging = 0.1, 
             stop = 10, l1 = 0, l2 = 1, dropout = 0.1, skipdrop = 0.5, verbose = False, info = True):
    
    if noclass is None: 
        try: noclass = nunique(array(hcat(y_train,y_test)))
        except: noclass = nunique(array(vcat(y_train,y_test)))

    if gpu == True: gpu = "gpu"
    else: gpu = "cpu"
    if min_weight <1: min_weight = int(min_weight*(len(vcat(x_train,y_train))))
    if bagging != False: bagged = 1;
    else: bagged = 0;
    if verbose == True: verbose = 1;
    else: verbose = 0;
    leaves = int(leaves); min_weight = int(min_weight); bins = int(bins)
    
    try: import lightgbm as lgb
    except:
        print("Cannot import"); raise

    x_train = array(x_train); y_train = array(y_train); x_test = array(x_test); y_test = array(y_test)
    train_data = lgb.Dataset(x_train,label=y_train)

    mets = metrics(noclass,"lightgbm")
    param = {'num_leaves':leaves, 'application':mets[0],'max_depth':depth,'learning_rate':lr,'num_iterations':trees, 'device':gpu,
             'max_depth':depth, 'metric':mets[1],'min_sum_hessian_in_leaf':impurity,'feature_fraction':features,
            'min_data_in_bin':bins,'bagging_fraction':bagging,'bagging_freq':bagged,'early_stopping_round':stop,'lambda_l1':l1,
            'lambda_l2':l2,'verbose':verbose,'nthread':jobs}
    
    if method == "dart": param['drop_rate'] = dropout; param['skip_drop'] = skipdrop
    elif mets[1] == 'multiclass': param['num_class'] = noclass
        
    print("--------------------------------\nLightGBM: Training...")
    modeller=lgb.train(param,train_data,trees)
    print("Finished")
    
    if info == True:
        if mets[0] == ('binary' or 'multiclass'): preds = toclasses(modeller.predict(x_test), unique(hcat(y_train,y_test)))
        else: preds = modeller.predict(x_test)
        for k in list(mets[2].keys()): 
            if k != 'rmse': print("Score = "+str(k)+" = "+str(mets[2][k](y_test, preds)))
            else: print("Score = "+str(k)+" = "+str(mets[2][k](y_test, preds)**0.5))
                
    return modeller

#-------------------- RF --------------------#
def randomforest(x_train, x_test, y_train, y_test, noclass = None, lr = 0.05, method = "dart", gpu = False, trees = 100, metric = None, 
             depth = -1, splits=2, leaves=31.123, min_weight=20, features=1, bins=5.123, impurity=1e-3+0.000001, jobs=-1, state=None, bagging = 0.1, 
             stop = 10, l1 = 0, l2 = 1, dropout = 0.1, skipdrop = 0.5, verbose = False, info = True, addon = False):
    
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    if noclass is None: 
        try: noclass = nunique(array(hcat(y_train,y_test)))
        except: noclass = nunique(array(vcat(y_train,y_test)))
            
    if depth == -1: depth = None;
    if method not in ["gini","entropy"]: method = "gini";
    if features == 1: features = "auto";
    if impurity == (1e-3+0.000001): impurity = 1e-07;
    if leaves == 31.123: leaves = None;
    if min_weight == 20.123: min_weight = 0;
    if bins == 5.123: bins = 1;

    leaves = int(leaves); bins = int(bins)
    
    x_train = array(x_train); y_train = array(y_train); x_test = array(x_test); y_test = array(y_test)
    mets = metrics(noclass,"randomforest")
    
    if mets[0] != 'regression':
        modeller = RandomForestClassifier(n_estimators=trees, criterion=method, max_depth=depth, min_samples_split=splits, min_samples_leaf=bins, 
                                          min_weight_fraction_leaf=0.0, max_features=features, max_leaf_nodes=leaves, min_impurity_split=impurity, 
                                          bootstrap=True, oob_score=info, n_jobs=jobs, random_state=state, verbose=verbose, warm_start=addon)
    else:
        modeller = RandomForestRegressor(n_estimators=trees, criterion="mse", max_depth=depth, min_samples_split=splits, min_samples_leaf=bins, 
                                          min_weight_fraction_leaf=0.0, max_features=features, max_leaf_nodes=leaves, min_impurity_split=impurity, 
                                          bootstrap=True, oob_score=info, n_jobs=jobs, random_state=state, verbose=verbose, warm_start=addon)
    
    print("--------------------------------\nRandomForest: Training...")
    modeller.fit(x_train,y_train)
    print("Finished")
    
    if info == True:
        preds = modeller.predict(x_test)
        for k in list(mets[1].keys()): 
            if k != 'rmse': print("Score = "+str(k)+" = "+str(mets[1][k](y_test, preds)))
            else: print("Score = "+str(k)+" = "+str(mets[1][k](y_test, preds)**0.5))
        print("Score = "+"OOB"+" = "+str(modeller.oob_score_))
    return modeller

#-------------
#-------------
#-------------
#-------------
#------------------------------------ SCALING AND NORMALISING ------------------------------------#
#-------------------- Standardise --------------------#
def standardise(data, output = True, method = "robust"):
    if method == "robust": from sklearn.preprocessing import RobustScaler as scaler
    elif method == "standard": from sklearn.preprocessing import StandardScaler as scaler
    elif "min" in method or "max" in method: from sklearn.preprocessing import MinMaxScaler as scaler
    elif "abs" in method: from sklearn.preprocessing import MaxAbsScaler as scaler
    
    if type(data)==pd.DataFrame: cols = columns(data)
    scaler = scaler()
    res = scaler.fit(data)
    res = scaler.transform(data)
    if type(data)==pd.DataFrame: 
        res = table(res)
        res.columns = cols
    if output == True: return res, scaler
    else: return res
 
#-------------------- Normalise --------------------#   
def normalise(data, output = True, method = "l2"):
    from sklearn.preprocessing import Normalizer
    
    if type(data)==pd.DataFrame: cols = columns(data)
    scaler = Normalizer(norm=method).fit(data)
    res = scaler.transform(data)
    if type(data)==pd.DataFrame: 
        res = table(res)
        res.columns = cols
    if output == True: return res, scaler
    else: return res

#-------------
#-------------
#-------------
#-------------
#------------------------------------ PREPROCESS FUNCTION ------------------------------------#
#-------------------- :) --------------------#

def preprocess(train, target, hold = 0.2, dummy = True, impute = "bpca", mice = "boost",remove_outlier = 0, scale = "robust", transform = 0,
              norm = False, output = True):
    
    processor = {'dummies':-1, 'impute':-1, 'scale':-1, 'transform':-1, 'norm':-1, 'columns':-1}
    
    if remove_outlier == 1: train = notoutlier(train)
        
    if type(target)==str:  x = exc(train, target);  y = train[target]
    if nunique(y)<=15: processor['target'] = unique(y)
    else: processor['target'] = -1
        
    x = complete(x, method = impute, mice = mice)
    
    if transform == (1 or True): x, unskewer = unskew(x, info = True)
    
    if dummy == False: x, codes = dummies(x, dummies = dummy, codes = True, ids = True)
    else: x = dummies(x, dummies = dummy, ids = True); codes = -2
    
    x = conts(x)
    if scale is not None and scale != False:  
        if scale == True: x, scaler = standardise(x, method = "robust")
        else: x, scaler = standardise(x, method = scale)
    
    if norm is not None and norm != False: 
        if norm == True: x, normer = normalise(x, method = "l2")
        else: x, normer = normalise(x, method = norm)
    
    if hold != (0 or False) and hold is not None: x_train, x_test, y_train, y_test = holdout(x, y = y)
    
    print("Processing finished :)")
    if output == True:
        try: processor['dummies'] = codes
        except: pass;
        try: processor['impute'] = [impute,train,mice]
        except: pass;
        try: processor['scale'] = scaler
        except: pass;
        try: processor['norm'] = normer
        except: pass;
        try: processor['transform'] = unskewer
        except: pass;
        processor['columns'] = columns(x_train)
        return x_train, x_test, y_train, y_test, processor
    
    else: return x_train, x_test, y_train, y_test

#-------------------- :) Transform the test data --------------------#
def prefit(test, processor):
    alldf = reset(vcat(processor['impute'][1],test), drop = True)
    df =  complete(alldf, method = processor['impute'][0], ids = True, mice = processor['impute'][2])

    test = df[len(processor['impute'][1]):]
    if processor['dummies'] == -2:  test = dummies(test, dummies = True, ids = True)
    a = set(processor['columns'])
    b = set(columns(test))
    matching = set.intersection(a,b)
    not_matching = a.symmetric_difference(matching)
    test = test[list(matching)]
        
    if processor['dummies'] == -2: 
        try: 
            tabs = int(table(np.zeros((len(test),len(not_matching)))))
            tabs.columns = list(not_matching)
            test[columns(tabs)] = tabs
        except: pass;
        test = test[processor['columns']]
    else:
        for key in list(processor['dummies'].keys()):
            try: test[key] = test[key].replace(processor['dummies'][key])
            except: pass;
        test = conts(test)

    if processor['scale']!=-1: test = processor['scale'].transform(test)
    if processor['norm']!=-1: test = processor['norm'].transform(test)
        
    df = table(test)
    df.columns = processor['columns']
    return df

#-------------
#-------------
#-------------
#-------------
#------------------------------------ METRICS AND HOLDOUT ------------------------------------#

def holdout(x, y, test = 0.2, seed = 1234, info = True):
    from sklearn.model_selection import train_test_split
    if info == True: print("--------------------------------\nx_train, x_test, y_train, y_test")
    return train_test_split(x, y, test_size = test, random_state = seed)
#--------------------
def metrics(noclass, model = "lightgbm"):
    from sklearn.metrics import mean_squared_error, cohen_kappa_score, r2_score
    
    if model == "lightgbm":
        if noclass == 2: return ['binary', ['binary_logloss','auc'], {'kappa':cohen_kappa_score,'rmse':mean_squared_error}]
        elif noclass < 15: return ['multiclass', ['multi_logloss','multi_error'], {'kappa':cohen_kappa_score,'rmse':mean_squared_error}]
        else: return ['regression_l2', ['l2_root'], {'r2':r2_score,'rmse':mean_squared_error}]
        
    elif model == "randomforest":
        if noclass == 2: return ['binary', {'kappa':cohen_kappa_score,'rmse':mean_squared_error}]
        elif noclass < 15: return ['multiclass', {'kappa':cohen_kappa_score,'rmse':mean_squared_error}]
        else: return ['regression', {'r2':r2_score,'rmse':mean_squared_error}]
#--------------------        
def toclasses(preds, classes):
    preds = np.round(preds)
    preds = int(squeeze(preds, lower = min(classes), upper = max(classes)))
    return preds
#--------------------
def predict(test, model, processor):
    preds = model.predict(array(test))
    if processor['target'] != -1: return toclasses(preds, classes = processor['target'])
    else: return preds

#-------------
#-------------
#-------------
#-------------
#------------------------------------ GRAPHING ------------------------------------#
def plot(x = None, y = None, z = None, hue = None, size = 8, data = None, color = 'afmhot', smooth = True, n = 4):
    dfdf = copy(data)
    import matplotlib.pyplot as plt
    if data is None and x is not None: print("Need to specify data"); return
    try:
        if type(x)==str: xlabel = x; x = dfdf[xlabel]; x = dummies(x, dummies = False)
    except: pass;
    try: 
        if type(y)==str: ylabel = y; y = dfdf[ylabel]; y = dummies(y, dummies = False)
    except: pass;
    try: 
        if type(z)==str: zlabel = z; z = dfdf[zlabel]; z = dummies(z, dummies = False)
    except: pass;
    try: 
        if type(hue)==str: huelabel = hue; hue = dfdf[huelabel]; hue = dummies(hue, dummies = False)
    except: pass;
    try: 
        xlabel = columns(x)[0]; 
        if xlabel is None: xlabel = "X"
    except: pass;
    try:
        ylabel = columns(y)[0]; 
        if ylabel is None: ylabel = "Y"
    except: pass;
    try: 
        zlabel = columns(z)[0];
        if zlabel is None: zlabel = "Z"
    except: pass;
    try: 
        huelabel = columns(hue)[0]; 
        if huelabel is None: huelabel = "Hue"
    except: pass;

    if x is not None and y is not None and z is not None:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        fig = plt.figure(figsize=(size,size))
        ax = Axes3D(fig)
        if hue is not None:
            cm = plt.get_cmap(color)
            try: cNorm = matplotlib.colors.Normalize(vmin=np.min(hue)[0], vmax=np.max(hue)[0])
            except: cNorm = matplotlib.colors.Normalize(vmin=np.min(hue), vmax=np.max(hue))
            scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)
            ax.scatter(array(x),array(y),array(z),c=scalarMap.to_rgba(array(hue)),s=size*5)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
            scalarMap.set_array(hue)
            fig.colorbar(scalarMap, pad=0, orientation = "h", shrink = .8)
            plt.show()
        else:
            import matplotlib
            ax.scatter(x,y,z,s=size*5)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
            plt.show()
    
    else:
        import seaborn as sb
        try: 
            if check_type(dfdf[xlabel]) == 'cat': dfdf[xlabel] = string(dfdf[xlabel])
        except: pass;
        try: 
            if check_type(dfdf[ylabel]) == 'cat': dfdf[ylabel] = string(dfdf[ylabel])
        except: pass;
        try: 
            if check_type(dfdf[huelabel]) == 'cat': dfdf[huelabel] = string(dfdf[huelabel])
        except: pass;

        if y is None and hue is None:
            fig = plt.figure(figsize=(size,size))

            if check_type(dfdf[xlabel]) == 'cont':
                fig = sb.kdeplot(data = dfdf[xlabel], linewidth = 3,clip = [min(dfdf[xlabel]),max(dfdf[xlabel])])
                mean_line(dfdf[xlabel])
                plt.ylabel("Frequency"); plt.xlabel(xlabel); plt.title("Kernel Density graph"); plt.show()

            elif check_type(dfdf[xlabel]) == 'cat':
                fig = sb.countplot(dfdf[xlabel].fillna("Missing"))
                plt.title("Count graph for "+xlabel); plt.show()

        elif y is None:
            if check_type(dfdf[xlabel]) == 'cont': sort_by = xlabel
            else: sort_by = huelabel

            if dtypes(dfdf[huelabel])[0] != 'obj':
                df = sort(dfdf, by = sort_by)
                dfdf[sort_by+"_Q"] = qcut(dfdf[sort_by], smooth = smooth, n = n)
                dfdf[sort_by+"_Q"] = string(dfdf[sort_by+"_Q"])

            fig = plt.figure(figsize=(size,size))
            if check_type(dfdf[xlabel]) == 'cont':

                if check_type(dfdf[huelabel]) == "cont":
                    fig = sb.violinplot(x=xlabel+"_Q", y=huelabel, bw='scott' ,scale="width",
                                        cut=min(dfdf[huelabel]), inner = None, linewidth =4, data = dfdf)
                    plt.setp(fig.get_xticklabels(), rotation=45);    plt.title("Violin graph for "+xlabel+" & "+huelabel)
                    plt.show()

                elif check_type(dfdf[huelabel]) == 'cat':
                    fig = sb.countplot(x = xlabel+"_Q", hue = huelabel, data = dfdf)
                    plt.title("Count graph for "+xlabel+" & "+huelabel); plt.setp(fig.get_xticklabels(), rotation=45)
                    plt.show()

            elif check_type(dfdf[xlabel]) == 'cat':
                if check_type(dfdf[huelabel]) == "cont":
                    fig = sb.countplot(x = xlabel, hue = huelabel+"_Q", data = dfdf)
                    plt.title("Count graph for "+xlabel+" & "+huelabel); plt.setp(fig.get_xticklabels(), rotation=45)
                    plt.show()

                if check_type(dfdf[huelabel]) == "cat":
                    fig = sb.countplot(x = xlabel, hue = huelabel, data = dfdf)
                    plt.title("Count graph for "+xlabel+" & "+huelabel); plt.setp(fig.get_xticklabels(), rotation=45)
                    plt.show()

        elif hue is None:
            if check_type(dfdf[xlabel]) == 'cont':

                if check_type(dfdf[ylabel]) == 'cont':
                    fig = plt.figure(figsize=(size,size))
                    dfdf = notnull(dfdf)
                    dfdf[xlabel+"_Q"] = qcut(dfdf[xlabel], n = 30, smooth = True)
                    dfdf = (dfdf.groupby(by = xlabel+"_Q").median()+dfdf.groupby(by = xlabel+"_Q").mean())/2
                    sb.regplot(x = xlabel, y = ylabel, data = dfdf, ci = None, truncate=True, order=2, color = 'black')
                    plt.title("Regression graph for "+xlabel+" & "+ylabel); plt.show()

                elif check_type(dfdf[ylabel]) == 'cat':
                    fig, (ax1,ax2) = plt.subplots(1,2, sharey = True, figsize = (size*1.5,size))
                    sb.boxplot(x = xlabel, y = ylabel, data = dfdf, palette="Set3", linewidth = 3, whis = 1, ax = ax1)
                    sb.pointplot(x = xlabel, y = ylabel, data = dfdf, lw=5, ax = ax2, ci = 50, capsize = .1, palette = 'Set1')
                    plt.title("Mean PointPlot graph for "+xlabel+" & "+ylabel); plt.show()

            elif check_type(dfdf[xlabel]) == 'cat':

                if check_type(dfdf[ylabel]) == 'cont':
                    fig, (ax1,ax2) = plt.subplots(1,2, sharey = False, figsize = (size*1.5,size))
                    sb.boxplot(x = xlabel, y = ylabel, data = dfdf, palette="Set3", linewidth = 3, whis = 1, ax = ax1)
                    plt.setp(ax1.get_xticklabels(), rotation=45)
                    plt.setp(ax2.get_xticklabels(), rotation=45)
                    sb.pointplot(x = xlabel, y = ylabel, data = dfdf, lw=5, ax = ax2, ci = 50, capsize = .1, palette = 'Set1')
                    plt.title("Mean PointPlot graph for "+xlabel+" & "+ylabel); plt.show()

                elif check_type(dfdf[ylabel]) == 'cat':
                    fig = sb.factorplot(x = xlabel, col = ylabel, data = dfdf, size = 5, palette="Set2", col_wrap = 4, kind = "count")
                    plt.show()        
        else:
            if check_type(dfdf[huelabel]) == 'cont':
                dfdf = notnull(sort(dfdf, by = huelabel))
                dfdf[huelabel] = string(qcut(dfdf[huelabel], smooth = False, n = 4))

            elif check_type(dfdf[huelabel]) == 'cat':
                dfdf = notnull(dfdf)

            if check_type(dfdf[xlabel]) == 'cat':

                if check_type(dfdf[ylabel]) == 'cont':
                    try: 
                        fig = plt.figure(figsize=(size,size))
                        fig = sb.barplot(x = xlabel, y = ylabel, hue = huelabel, data = dfdf)
                        plt.setp(fig.get_xticklabels(), rotation=45)
                        plt.show()
                    except:
                        fig = sb.factorplot(x = xlabel, y = ylabel, data = dfdf, col = huelabel, size = 5, capsize=.1, palette="Set2", ci = 70)
                        plt.show()

                elif check_type(dfdf[ylabel]) == 'cat':
                    fig = sb.factorplot(x = xlabel, hue = ylabel, data = dfdf, col = huelabel, kind = "count", size = 5)
                    plt.show()

            elif check_type(dfdf[xlabel]) == 'cont':

                if check_type(dfdf[ylabel]) == 'cont':
                    fig = plt.figure(figsize=(size,size))
                    fig = sb.lmplot(x = xlabel, y = ylabel, hue = huelabel, data = dfdf,robust = True, n_boot = 50, scatter = False, ci = None)
                    plt.show()

                elif check_type(dfdf[ylabel]) == 'cat':
                    fig = sb.factorplot(x = xlabel, y = ylabel, col = huelabel, data = dfdf, palette = "Set3", dodge=True, ci = 70, 
                                        estimator = special_statistic, capsize=.2, n_boot = 100, size = 5)
                    plt.show()
            
def highlight_larger(s):
    is_max = s > CI(99,s,L=False);    return ['background-color: '+pd_colour if v else '' for v in is_max]
def highlight_smaller(s):
    is_min = s < CI(99,s,U=False);    return ['background-color: '+pd_colour if v else '' for v in is_min]
def highlight_one(s):
    is_true = s == 1;                 return ['background-color: '+pd_colour if v else '' for v in is_true]
def highlight_true(s):
    is_true = s == True;              return ['background-color: '+pd_colour if v else '' for v in is_true]

#-------------            
def mean_line(x, **kwargs):
    ls = {"0":"--"}
    plt.axvline(mean(x), linestyle =ls[kwargs.get("label","0")], 
                color = kwargs.get("color", "brown"), linewidth=2)
    txkw = dict(size=12, color = kwargs.get("color", "brown"))
    plt.text(mean(x),0.03, "MEAN", **txkw)
#-------------    
def special_statistic(x): return (2*np.nanmedian(x)+np.nanmean(x))/3
#-------------
def check_type(x):
    ctd = nunique(x);         parts = (((ctd<=15)&(len(x)>15))|((ctd<len(x)*0.01)&(ctd<=20)&(dtypes(x)=='int'))|((dtypes(x)=='str')&(ctd<=15)))
    if dtypes(x) != 'str':
        if parts == True: return 'cat'
        else: return 'cont'
    else: 
        if parts == False: return 'str'
        else: return 'cat'

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DATA MINING AND NLP ------------------------------------#
#-------------------- String extracting --------------------#
def getfunction(c, args, now):
    if "split" in c:
        if "ex" in c: expanding = True; 
        else: expanding = False
        if "letter" in args[c].lower() or "word" in args[c].lower() or "digit" in args[c].lower() or "number" in args[c].lower():
            how = ''
            for j in args[c].split(","):
                if "letter" in j: how = how+"([a-z])"
                elif "Letter" in j: how = how+"([a-zA-Z])"
                elif "LETTER" in j: how = how+"([A-Z])"
                elif "word" in j: how = how+"([a-z]+)"
                elif "Word" in j: how = how+"([a-zA-Z]+)"
                elif "WORD" in j: how = how+"([A-Z]+)"
                elif "digit" in j.lower(): how = how+"([0-9])"
                elif "number" in j.lower(): how = how+"([0-9]+)"
                elif "symbol" in j.lower():  how+'[^\w]+'
            now = now.str.extract(how, expand = expanding)
        else: now = now.str.split(args[c], expand = expanding)
    elif "col" in c or "loc" in c: 
        try: 
            if "le" in args[c]: now = now.str[0:-1]
            elif "ri" in args[c]: now = now.str[-1:1]
        except:
            if type(now) == pd.Series: now = now.str[args[c]]
            else: now = now[args[c]]
    elif "not" in c: now = now.str.contains(args[c]); now = reverse(now)
    elif "has" in c: now = now.str.contains(args[c])
    elif "rep" in c: 
        if "symbol" in args[c]: now = now.replace(r'[^\w]',args[c][1])
        else: now = now.str.replace(args[c][0], args[c][1])
    elif "rem" in c or "strip" in c:
        if "all" in args[c]:
            for j in [".",",","+","=","-","_","(",")","[","]","*","$","?","<",">",'"',"'","/","<",">","%"]:
                now = now.str.replace(j,"")
        elif "symbol" in args[c]: now = now.replace(r'[^\w]','')
        else: now = now.str.replace(args[c][0], "")
    elif "len" in c: 
        if args[c] == 1: now = now.str.len()
    elif "low" in c:
        if args[c] == 1: now = now.str.lower()
    elif "up" in c:
        if args[c] == 1: now = now.str.upper()
    elif "count" in c: 
        if args[c] == ".": now = now.str.count(r"(\.)")
        elif args[c] == "(": now = now.str.count(r"(\()")
        elif args[c] == ")": now = now.str.count(r"(\))")
        elif args[c] == "[": now = now.str.count(r"(\[)")
        elif args[c] == "]": now = now.str.count(r"(\])")
        elif args[c] == "{": now = now.str.count(r"(\{)")
        elif args[c] == "}": now = now.str.count(r"(\})")
        elif 'symbol' in args[c]: now = now.str.count(r'[^\w]')
        elif 'sym' in args[c]: now = now.str.count(r'[\w]')
        elif 'num' in args[c] or 'dig' in args[c]: now = now.str.count(r'[\d]')
        else: now = now.str.count(args[c]) 
    elif "df" in c or "table" in c or "series" in c: now = now.apply(pd.Series)
    return now
                 
def get(x, **args):
    import re
    now = copy(x)
    for c in args:
        now = getfunction(c, args, now)
    return now

def extract(x, **args): return get(x, args)
#-------------------- Word Frequency --------------------#
def flatten(y, split = " ", dropna = True, symbols = False, lower = True):
    
    def col_split(x,split,dropna,symbols,lower):
        if split is not None: 
            if symbols == False: 
                if lower == True: f = list(get(x, lower = True, rem = "all", splitex = split).fillna(np.nan).values.flatten())
                else: f = list(get(x, rem = "all", splitex = split).fillna(np.nan).values.flatten())
            else: f = list(get(x, splitex = split).fillna(np.nan).values.flatten())
        else: f = list(x.fillna(np.nan).values.flatten())
        return f
    
    if type(y)==pd.Series: flattened = col_split(y,split,dropna,symbols,lower)
    else:
        flattened = []
        for col in strs(y): 
            flattened += col_split(y[col],split,dropna,symbols,lower)
    if dropna == True: return list(array(flattened)[array(flattened)!='nan'])
    else: return flattened
#-------------    
def wordfreq(x, hist = True, first = 15, separate = True):
    if separate == False or type(x) == pd.Series:
        df = reset(table(cunique(flatten(x))))[0:first]
        df.columns = ["Word","Count"]
        
    else:
        first = int(first/len(strs(x)))
        df = reset(table(cunique(flatten(x[strs(x)[0]]))))[0:first]
        df.columns = ["Word","Count"]
        df["Column"] = objcol(x)[0]
        for col in objcol(x)[1:]:
            dfx = reset(table(cunique(flatten(x[col]))))[0:first]
            dfx.columns = ["Word","Count"]
            dfx["Column"] = col
            df = vcat(df,dfx)
            
    if hist == True: 
        
        if separate == True and type(x) != pd.Series:
            k = first*1.25
            if k < 10: k = 8
            fig = plt.figure(figsize=(k,k))
            fig = sb.barplot(x = "Word", y = "Count", hue = "Column", data = df)
            plt.setp(fig.get_xticklabels(), rotation=45, size = 16)
        else: 
            fig = plt.figure(figsize=(first*0.5,first*0.35))
            fig = sb.barplot(x = "Word", y = "Count", data = df)
            plt.setp(fig.get_xticklabels(), rotation=45, size = 16)
        plt.show()        
    else:
        return df
#-------------     
def getwords(y, first = 10):
    x = copy(y)
    df = wordfreq(x, first = first, hist = False)
    for col in objcol(x):
        cols = get(x[col], lower = True, rem = "all", table = True)
        for j in df[df["Column"]==col]["Word"]:
            x["Count="+str(j)] = get(cols[0], count = j)
    return x

#-------------

#------------- Daniel Han-Chen 2017
#------------- https://github.com/danielhanchen/sciblox
#------------- SciBlox v0.2.1
#-------------