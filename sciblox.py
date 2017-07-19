#------------- Daniel Han-Chen 2017
#------------- https://github.com/danielhanchen/sciblox
#------------- SciBlox v0.01
#-------------

# %matplotlib inline
# from jupyterthemes import jtplot
# jtplot.style()

import numpy as np
import pandas as pd
import scipy as sci
import scipy
import seaborn as sb
import matplotlib.pyplot as plt

import warnings
import sklearn as sk
warnings.filterwarnings("ignore")
np.set_printoptions(suppress = True)
pd.set_option('display.max_rows', 15)
true = True; TRUE = True
false = False; FALSE = False
pd_colour = '#302f2f'

def maxrows(x): pd.set_option('display.max_rows', x)
def tabcolour(x = 'black'): pd_colour = x
    

#------------------------------------ MATHEMATICAL METHODS ------------------------------------#
#-------------------- Arrays and matrix formation --------------------#
def array(*args): 
    if len(args) <=1: return np.array(args[0])
    else: return vcat(args)
#-------------    
def matrix(*args):
    from io import StringIO
    if type(args[0]) == str:
        text = args[0].replace(" ",",").replace("\\","\n")
        for j in range(30): text = text.replace(","*(30-j+1),"\n")
        text = StringIO(text)
        return np.matrix(pd.read_csv(text,",", header = None))
    else: return np.matrix(array(args))
    
def M(*args): return matrix(args)
def m(*args): return matrix(args)

#-------------------- Mathemaatical functions --------------------#
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
    try: return x.sum()
    except: return np.nansum(x, axis = 0)
#-------------    
def round(x, n = 2):
    return np.round(x, n)
#------------- 
def max(x, axis = 0):
    if type(x) in [pd.Series, pd.DataFrame]: return x[conts(x)].max()
    else:
        if shape(matrix(x))[0] == 1: return np.amax(x,axis=axis)
        else: return np.amax(x)
        
def min(x, axis = 0):
    if type(x) in [pd.Series, pd.DataFrame]: return x[conts(x)].min()
    else:
        if shape(matrix(x))[0] == 1: return np.amin(x)
        else: return np.amin(x,axis=axis)
        
#-------------------- Linear Algebra --------------------#        
def dot(x, y):
    try: return np.dot(x,y)
    except: 
        try: return np.dot(array(x),array(y))
        except: print("X has shape "+str(shape(x))+" whilst Y "+str(shape(y)))
            
def mult(x, y): return np.multiply(x, y)
#------------- 
def T(x):
    if ",)" in str(shape(x)) and type(x)!= pd.Series: return matrix(x).T
    elif type(x) == pd.Series: return table(x).T
    else: return x.T
    
def t(x): return T(x)
#------------- 
def inv(x):
    try: return np.inv(x)
    except: print("Either det(x)=0 or not square matrix")
        
def det(x):
    try: return np.det(x)
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
def tr(A): return np.trace(A)
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
#------------------------------------ DATA MANIPULATION AND SELECTION ------------------------------------#
#-------------------- Data opening --------------------#
def read(x):
    if type(x) == list:
        for y in x: 
            if "csv" in y: return clean(pd.read_csv(y))
    else:
        if "csv" in x: return clean(pd.read_csv(x))

#-------------------- Basic Data Exploration and Data Information Accessing --------------------#
def table(x): 
    try: return pd.DataFrame(x)
    except: return pd.DataFrame(list(x.items()))
    
def DF(x): return pd.DataFrame(x)
def series(x): return pd.Series(x)
#------------- 
def copy(x): 
    import copy
    return copy.copy(x)
#-------------
def shape(x): 
    try: return x.shape
    except: return len(x)
#-------------
def head(x, n = 5): 
    if type(x) in [pd.DataFrame,pd.Series]: return x.head(n)
    else: 
        try:  return x[0:n]
        except: return x
        
def tail(x, n = 5): 
    if type(x) in [pd.DataFrame,pd.Series]: return x.tail(n)
    else: 
        try:  return x[-n:]
        except: return x
#------------- 
def sample(x, size = 5, n = 5, order = False, ordered = False):
    return random(x,size,n,order,ordered)

def random(x, size = 5, n = 5, order = False, ordered = False):
    if type(x) in [pd.DataFrame,pd.Series]:
        nn = 5
        if size != 5: nn = size
        if n != 5: nn = n
        if order != False or ordered != False:
            to_get = [int(y*len(x)/nn) for y in range(nn)][1:]
            try:
                df = x.iloc[[0]]
                for to in to_get: df = vcat(df, x.iloc[[to]])
            except:
                df = x.iloc[0]
                for to in to_get: df = vcat(df, x.iloc[to])
            return df

        else: return pd.DataFrame.sample(x,n=nn)
    else: return np.random.choice(x, nn)
#-------------     
def columns(x):
    if type(x) in [pd.Series, pd.DataFrame]: 
        try:  return x.columns.tolist()
        except:  pass;
    else: print("Not DataFrame")
        
def cols(x): return columns(x)

def index(x):
    if type(x) in [pd.Series, pd.DataFrame]: return x.index.tolist()
    
#-------------------- Getting more advanced information from a dataset and transforming data: --------------------#
def conts(x):
    if type(x) == pd.DataFrame: return (x.dtypes!="O").index[x.dtypes!="O"].tolist()
    elif type(x) == pd.Series: return (pd.DataFrame.transpose(table(x)).dtypes!="O").index[pd.DataFrame.transpose(table(x)).dtypes!="O"].tolist()

def strs(x):
    if type(x) == pd.DataFrame: return (x.dtypes=="O").index[x.dtypes=="O"].tolist()
    elif type(x) == pd.Series: return (pd.DataFrame.transpose(table(x)).dtypes=="O").index[pd.DataFrame.transpose(table(x)).dtypes=="O"].tolist()
    
def cats(x):
    if type(x) == pd.DataFrame: 
        objects = strs(x);        ctd = nunqiue(x[conts(x)]);         parts = ((ctd<6)|((ctd<len(x)*0.01)&(ctd<15)&(dtypes(x[conts(x)])=='int')))
        categories = parts.index[parts].tolist()                  
        for cat in categories: objects.append(cat)
        return objects
    elif type(x) == pd.Series: return (pd.DataFrame.transpose(table(x)).dtypes=="O").index[pd.DataFrame.transpose(table(x)).dtypes=="O"].tolist()
    
def objects(x): return strs(x)
def objs(x): return strs(x)
#-------------    
def string(df, *args):
    if type(df) == pd.DataFrame:
        x, col = argcheck(df, args)
        for y in col:
            x[y] = x[y].astype("str")+"*"
        return x
    elif type(df) == pd.Series: 
        df = df.astype("str")+"*"
        return df
    else: return str(df)
    
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
def dtypes(x, *args):
    if len(args)==0: col = columns(x)
    elif type(args[0])!=list: col = list(args)
    else: col = args[0]
    if type(x) == pd.DataFrame:
        df = x[col].dtypes
        df[(df == "int64") | (df=="int32")] = "int"
        df[(df == "float64") | (df=="float32")] = "float"
        df[(df == "object")] = "str"
        return df
    elif type(x) == pd.Series:
        if x.dtype in ["int64","int32"]: return "int"
        elif x.dtype in ["float32","float64"]: return "float"
        elif x.dtype == 'bool': return 'bool'
        else: return "str"
    else: return type(x)
            
#-------------
def missing(x): return (x.count()!=len(x)).index[x.count()!=len(x)].tolist()
def nacols(x): return (x.count()!=len(x)).index[x.count()!=len(x)].tolist()

def isnull(x, axis = 1):
    if axis == 1: 
        if type(x) == pd.DataFrame: return x[x.isnull().any(axis=1)]
        else: return x[x.isnull()]
    else: return x[missing(x)]
    
def notnull(x, axis = 0, subset = None):
    if type(x) == pd.DataFrame: return x.dropna(axis = axis, subset = None)
    else: return x.dropna()
        
def count(x): return x.count()
def counts(x): return count(x)

def zerocount(x): return sum(x==0)
#-------------------- Cleaning some data and popping/joining columns / getting specific columns --------------------#
def clean(x, *args):
    def cleancol(x):
        if dtypes(x) == 'obj': 
            c = x.str.replace(",","").str.replace(" ","").str.replace("-","").str.replace("%","").str.replace("#","")
        else: c = x
            
        try: 
            if ((sum(int(c)) - sum(float(c)) == 0) or sum(int(c)-float(c))==0) and count(c) == len(c): return int(c)
            else: return float(c)
        except: 
            return c
            
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
def shiftup(x): return x.apply(lambda x: pd.Series(x.dropna().values))
#-------------       
def pop(x, *args):
    if type(args[0]) != list: obj = list(args)
    else: obj = args[0]
    if type(x) == pd.DataFrame:
        if type(obj) == list:
            k = {}
            for j in obj: k[j].append(x.pop(j))
            return k
        else: return x.pop(obj)
    elif type(x) == list:
        for g in obj:
            try: x.remove(g)
            except: pass
        return x
    else: 
        if type(obj[0]) == str: 
            if dtypes(x)!= 'str': x = string(x)
                
def rem(x, *args): return pop(x, args)
def remove(x, *args): return pop(x, args)
#-------------
def dropna(x, h):
    if type(h) == str: h = [h]
    return x.dropna(subset = h, how = "all")
#-------------
def exc(x, l):
    if type(l) == str: l = [l]
    return x[x.columns.difference(l)]

def inc(x, l):
    if type(l) == str: l = [l]
    return x[l]

def diff(want, rem):
    w = copy(want)
    for j in w:
        if j in rem: w.remove(j)
    return w

def append(left, right): 
    return left+right
#-------------
def reverse(x):
    if type(x) in [pd.DataFrame,pd.Series]: 
        if dtypes(x) == 'bool': return x == False
        else: return x.iloc[::-1]
    elif type(x) == list: return x[::-1]
    elif type(x) == dict: return {i[1]:i[0] for i in x.items()}
    
def rev(x): return reverse(x)
#-------------
def hcat(*args):
    first = args[0]
    if type(first) == pd.Series and len(args)>1: first = table(first)
    elif type(first) != pd.DataFrame: first = np.array(first)
    for x in list(args)[1:]: 
        if type(first) in [np.array,np.ndarray]: first = np.hstack((first, x))
        elif type(x) == type(args[0]) == pd.Series: first = pd.concat([first, table(x)], 1) 
        else: first = pd.concat([first, table(x)],1,ignore_index=True)
    return first
    
def vcat(*args):
    first = args[0]
    if type(first) == pd.Series and len(args)>1: first = table(first)
    elif type(first) != pd.DataFrame: first = np.array(first)
    for x in list(args)[1:]: 
        if type(first) in [np.array,np.ndarray]: first = np.vstack((first, x))
        elif type(x) == type(args[0]) == pd.Series: first = pd.concat([first, table(x)]) 
        else: first = first.append(table(x))
    return first
    
#-------------------- Sorting and Frequencies --------------------#
def nunique(x): 
    if type(x) in [pd.DataFrame,pd.Series]: return x.nunique()
    else: return len(unique(x))
    
def nunqiues(x): return nunique(x)
def nuniques(x): return nunique(x)
def nunqiue(x): return nunique(x)
    
def unique(x):
    if type(x) == pd.Series: return x.unique()
    elif type(x) == pd.DataFrame:
        xdict = {}
        for y in columns(x): xdict[y] = x[y].unique().tolist()
        return xdict
    else:
        xlist = []
        for y in x: 
            if y not in xlist: xlist.append(y)
        return xlist

def unqiue(x): return unique(x)

def punique(x): return round(nunique(x)/len(x),3)
def punqiue(x): return punique(x)

def cunique(x, keep_na = False, *args):
    if type(x) == list: x = table(x)
    if type(x) == pd.Series: 
        if keep_na == False: return x.value_counts(dropna = True)
        else: return x.value_counts(dropna = False)
    else: 
        if len(args) == 0: col = columns(x)
        else: col = list(args)
        unqs = {}
        for c in col:
            if keep_na == False: unqs[c] = x[c].value_counts(dropna = True)
            else: unqs[c] = x[c].value_counts(dropna = False)
        return unqs
#-------------
def reset(x, index = True): 
    if index == True: return x.reset_index()
    else:
        x.columns = [y for y in range(len(columns(x)))]
        return x
#-------------
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
#-------------            
def fsort(x, by = None, keep = False, asc = True, ascending = True, des = False, descending = False):
    if type(x) == list:
        from collections import Counter
        c = copy(x)
        if asc == ascending == True and des == descending == False: c.sort(key=Counter(sort(c, asc = True)).get, reverse = True); return c
        else: c.sort(key=Counter(sort(c, asc = False)).get, reverse = False); return c
    elif by is None: print("Please specify 1 column")
    else:
        f = by; fg = table(x[f].value_counts()).reset_index()
        ff = f+"_Freq"; fg.columns = [f,ff]
        
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
#------------------------------------ DATA DESCRIPTION AND STATISTICS ------------------------------------#
#-------------------- Statistics --------------------#
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
    return max(x, axis = axis) - min(x, axis = axis)
#-------------
def percentile(x, p, axis = 0):
    if p > 1: p = p/100
    try: return x.quantile(p, axis = axis)
    except: return np.nanpercentile(x, p, axis = axis)

def iqr(x, axis = 0):
    return percentile(x, 0.75, axis = axis) - percentile(x, 0.25, axis = axis)
#-------------
def skewness(x, axis = 0):
    try: return sci.stats.skew(x, axis = axis, nan_policy='omit')
    except: return x.skew(p, axis = axis)
    
def skew(x, axis = 0): return skewness(x, axis)
    
def kurtosis(x, axis = 0):
    try: return sci.stats.kurtosis(x, axis = axis, nan_policy='omit')
    except: return x.kurt(p, axis = axis)
    
def kurt(x, axis = 0): return kurtosis(x, axis)
#-------------
def pnorm(p, mean = 0, var = 1):
    if p > 1: p = p/100
    return sci.stats.norm.cdf(p, loc=mean, scale=var)

def qnorm(q, mean = 0, var = 1):
    if q > 1: q = q/100
    return sci.stats.norm.ppf(q, loc=mean, scale=var)

def CI(q, data, method = "mean",U = True, L = True):
    if q > 1: q = q/100 ; norms = qnorm(q+(1-q)/2)*(std(data) / sqrt(len(data)) )
    if method == "mean": u = mean(data) + norms; l = mean(data) - norms
    if U == L == True: return (l,u)
    elif U == True: return u
    else: return l
#-------------------- Basic Analysis --------------------#
def describe(x, strings = True, continuous = True, axis = 0, colour = True):
    if type(x) == pd.Series:
        if dtypes(x) == 'str':
            things = table(vcat(mode(x), nunique(x), punique(x), freqratio(x)))
            things.index = ["Mode","No.Unique","%Unique","FreqRatio"]
            return things[0]
        else:
            things = round(table(vcat(mean(x), median(x), mode(x), rng(x), iqr(x), freqratio(x), var(x))),3)
            things.index = ["Mean", "Median", "Mode", "Range", "IQR", "FreqRatio", "Var"]
            return things[0]
    else: 
        dicts = {}
        if continuous == True:
            for col in conts(x):
                v = x[col];            dicts[col] = round([mean(v),median(v),rng(v),iqr(v),var(v),mode(v),freqratio(v),punique(v),nunqiue(v)])
                                                    
        if strings == True:
            for col in strs(x):
                v = x[col];            dicts[col] = ["","","","","",mode(v),round(freqratio(v)),round(punique(v)),round(nunqiue(v))]
        things = T(table(dicts))
        things.columns = ["Mean","Median","Range","IQR","Var","Mode","FreqRatio","%Unique","No.Unique"]
        if colour == True:
            dfxs = things.replace("",np.nan).style.bar(subset =  ["Mean","Median","Range","IQR","Var","FreqRatio","%Unique","No.Unique"],
                                     color=['#000000', pd_colour], align = "mid", axis = axis, width = 75)
            return dfxs.set_properties(**{'max-width': '90px'})
        else: return things
        
def summary(x, strings = True, continuous = True, axis = 0, colour = True): return describe(x,strings,continuous,axis,colour)
#-------------
def check(x):
    df = hcat(dtypes(x), round(((len(x)-count(x))/len(x)*100),0).astype("int"), round((zerocount(x)/len(x)*100),0).astype("int")  )
    df.columns = ["Type","%Missing","%Zeroes"]
    return df
#-------------
def analyse(x, colour = True, axis = 0):
    dt = describe(x, colour = False)
    dx = check(x)
    df = hcat(dx, dt).replace("",np.nan)
    df.columns = columns(dx)+columns(dt)
    if colour == True:
        dfxs = df.replace("",np.nan).style.bar(subset =  ["Mean","Median","Range","IQR","Var","FreqRatio","%Unique","No.Unique",
                                                         "%Missing","%Zeroes"],
                                 color=['#000000', pd_colour], align = "mid", axis = axis, width = 75)
        return dfxs.set_properties(**{'max-width': '90px'})
    else: return df
#-------------    
def freqratio(x, **arg):
    if type(x) == pd.DataFrame:
        l = []
        for col in columns(x): l.append(  round(  cunique(x[col]).iloc[0] / cunique(x[col]).iloc[1]  ))
        u = table(l).T
        u.columns = columns(x)
        if len(arg)!=0:
            if "t" in list(arg.keys())[0].lower(): return (u.T[0])[u.T[0]<=list(arg.values())[0]]
        return u.T[0]
    elif type(x) == pd.Series: 
        if len(arg)!=0:
            if "t" in list(arg.keys())[0].lower():
                u = round(  cunique(x).iloc[0] / cunique(x).iloc[1])
                return (u<=list(arg.values())[0])
        else: return round(  cunique(x).iloc[0] / cunique(x).iloc[1])
        
def varcheck(x, **args):
    j = 0
    if type(x) == pd.Series: x = table(x); j = 1
    rows = round(hcat(freqratio(x),punqiue(x),var(x),var(x)>0.001),3)
    rows.columns = ["FreqRatio", "%Unique", "Var","VarGood?"]
    if j == 1: rows = shiftup(rows)
    if len(args)==0: 
        rows = rows.style.apply(highlight_larger, subset=['FreqRatio']).apply(highlight_smaller, subset=['%Unique']).apply(nearzero, subset=['Var']).apply(false_bad, subset=['VarGood?'])
        return rows
    else: 
        for c in args:
            if "freq" in c.lower(): 
                old = columns(rows)
                if args[c] == 'mean': rows = hcat(rows, rows["FreqRatio"]<=CI(99, rows["FreqRatio"], L = False))
                else: rows = hcat(rows, rows["FreqRatio"]<=args[c])
                rows.columns = hcat(old,'FreqRatioGood?')
            if "un" in c.lower(): 
                old = columns(rows)
                if args[c] == 'mean': rows = hcat(rows, rows["%Unique"]>=CI(99, rows["%Unique"], U = False))
                elif args[c] > 1: args[c] = args[c]/100
                else: rows = hcat(rows, rows["%Unique"]>=args[c])
                rows.columns = hcat(old,'%UniqueGood?')
    try: old = columns(rows)
    except: rows = T(rows); old = columns(rows)
    try: 
        rows = hcat(rows, rows["%UniqueGood?"]*rows["FreqRatioGood?"]*rows["VarGood?"].fillna(True))
        rows.columns = hcat(old,'Good?')
        rows = rows.style.apply(highlight_larger, subset=['FreqRatio']).apply(highlight_smaller, subset=['%Unique']).\
                    apply(nearzero, subset=['Var']).apply(false_bad, subset=['VarGood?','%UniqueGood?',"FreqRatioGood?","Good?"])
    except:
        try: 
            rows = hcat(rows, rows["%UniqueGood?"]*rows["VarGood?"].fillna(True))
            rows.columns = hcat(old,'Good?')
            rows = rows.style.apply(highlight_larger, subset=['FreqRatio']).apply(highlight_smaller, subset=['%Unique']).\
                    apply(nearzero, subset=['Var']).apply(false_bad, subset=['VarGood?','%UniqueGood?',"Good?"])
        except: 
            try: 
                rows = hcat(rows, rows["FreqRatioGood?"]*rows["VarGood?"].fillna(True))
                rows.columns = hcat(old,'Good?')
                rows = rows.style.apply(highlight_larger, subset=['FreqRatio']).apply(highlight_smaller, subset=['%Unique']).\
                    apply(nearzero, subset=['Var']).apply(false_bad, subset=['VarGood?','FreqRatioGood?',"Good?"])
            except: 
                rows = hcat(rows, rows["VarGood?"].fillna(True))
                rows.columns = hcat(old,'Good?')
                rows = rows.style.apply(highlight_larger, subset=['FreqRatio']).apply(highlight_smaller, subset=['%Unique']).\
                    apply(nearzero, subset=['Var']).apply(false_bad, subset=['VarGood?',"Good?"])
    
    if "get" in list(args.keys()) or "col" in list(args.keys()) or "out" in list(args.keys()): return (rows["Good?"]==True).index[rows["Good?"]==True].tolist()
    else: return rows
#-------------    
def corr(x, table = False):
    if table == False:
        corrs = x.corr()
        cmap =sb.diverging_palette(100, 200, as_cmap=True)
        def magnify():
            return [dict(selector="th",
                         props=[("font-size", "11pt")]),
                    dict(selector="td",
                         props=[('padding', "0em 0em")]),
                    dict(selector="th:hover",
                         props=[("font-size", "16pt")]),
                    dict(selector="tr:hover td:hover",
                         props=[('max-width', '200px'),
                                ('font-size', '16pt')])
        ]
        show = corrs.style.background_gradient(cmap, axis=1).set_properties(**{'max-width': '100px', 'font-size': '11pt'}).set_precision(2).set_table_styles(magnify())
        return show
    else:
        try: return x[conts(x)].corr()
        except: print("Error. No continuous data")
            
def correlation(x, table = False): return corr(x, table)
def correlation_matrix(x, table = False): return corr(x, table)
def cor(x, table = False): return corr(x, table)
#-------------
## https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
def remcor(x, threshold = 0.9):
    dataset = copy(x)
    col_corr = set(); corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns: del dataset[colname]
    return dataset
#-------------
#https://stackoverflow.com/questions/42867494/how-can-i-find-a-basis-for-the-column-space-of-a-rectangular-matrix/42868363#42868363
def linindp(A):
    from scipy.linalg import lu
    U = lu(A)[2]
    col = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])]
    good = T(A)[0]
    for c in col[1:]: good = T(np.append(good,(t(A)[c]), axis=0))
    return good

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DATA CLEANING AND CONVERSION ------------------------------------#
#-------------------- Normal statistic filling --------------------#
def fillobj(x, method):
    data = copy(clean(x))
    missed = nacols(data[objs(data)]); missdf = data[missed]
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
    missed = nacols(data[conts(data)]); missdf = data[missed]
    if method in ["mean","avg","average"]: data[missed] = data[missed].fillna(mean(missdf))
    elif method in ["median"]: data[missed] = data[missed].fillna(median(missdf))
    elif method in ["mode","freq","frequency"]: data[missed] = data[missed].fillna(mode(missdf))
    return data
#-------------------- Full methods --------------------#
def fillna(df, method = None, objects = None, continuous = None, knn = 5, max_unique = 20, epoch = 100, mice = "forest"):
    x = copy(df);       imputation = ["bpca","pca","knn","mice","svd"];               imped = 0
    if method is not None: meth = method.lower()
    else: meth = "a"
    if method is None and objects is None and continuous is None: meth = 'knn'
        
    if meth in imputation or objects in imputation or continuous in imputation:
        imped = 1
        try: import fancyimpute
        except: print("Sorry, FancyImpute/Keras/MingW-C is not installed. Please install through Anaconda")
            
        def matching(method, objects, continuous, thingo):
            if method is not None:
                if thingo in method: return 1
                else: return 0
            else:
                if thingo in objects or thingo in continuous: return 1
                else: return 0
            
        codes, res = to_cont(x, class_max = max_unique, return_codes = True, dummies = False)
        
        if matching(meth, objects, continuous, "knn") == 1: dfilled = fancyimpute.KNN(k=knn).complete(res)
        elif matching(meth, objects, continuous, "svd") == 1: dfilled = fancyimpute.SoftImpute().complete(res)
        elif matching(meth, objects, continuous, "mice") == 1: 
            print("Please wait...")
            dfilled = mice_complete(res, epochs = int(epoch/10), impute_method = mice, strings = strs(x))
            print("Done")
        else: 
            print("Please wait...")
            dfilled = bpca_complete(res, epochs = epoch)
            print("Done")
            
        dfilled = table(dfilled); dfilled.columns = columns(res)

        for col in codes: x[col] = squeeze(series(int(round(dfilled[col],0))), upper = len(codes[col])-1, lower = 0).replace(reverse(codes[col]))
        x[conts(x)] = dfilled[conts(x)]
        for col in conts(x): x[col] = squeeze(x[col], lower = min(df[col]), upper = max(df[col]))

    if (missing(x) != [] and objects in imputation) or meth in imputation: x = fillobj(x, "mix")
    elif objects is not None: x[strs(x)] = fillobj(df[strs(df)], objects)
    
    if continuous not in imputation and continuous is not None: x[conts(x)] = fillcont(df[conts(df)], continuous)
    return x

#-------------------- BPCA --------------------#
#http://ishiilab.jp/member/oba/tools/BPCAFill.html
def bpca_complete(x, epochs = 100):
    y = copy(x); cols = columns(y)
    maximum = np.max(max(y))*9999
    means = y.mean();        sd = std(y);   y = (y-means)/sd
    y = y.fillna(maximum);   mat = float(np.matrix(y))

    def t(x): return np.transpose(np.matrix(x))

    N, d = shape(mat);      q = d-1
    yest = np.copy(mat);    yest[yest==maximum]=0

    missidx = {};       bad = np.where(mat==maximum)
    for x in range(N): missidx[x] = []
    for x in range(len(bad[0])): missidx[bad[0][x]].append(bad[1][x])

    nomissidx = {};     good = np.where(mat!=maximum)
    for x in range(N): nomissidx[x] = []
    for x in range(len(good[0])): nomissidx[good[0][x]].append(good[1][x])

    gmiss = sort(list(set(bad[0])))
    gnomiss = sort(list(set(np.where(np.sum(np.abs(mat),1)<maximum-10)[0])))

    covy = np.cov(t(yest))
    U, S, V = np.linalg.svd(np.matrix(covy))
    U = t(t(U)[0:q]);         S = S[0:q]*eye(q);           V = t(t(V)[0:q])

    mu = np.copy(mat);          mu[mu==maximum]=np.nan;        mu = np.nanmean(mu, 0)
    W = U*np.sqrt(S);         tau = 1/ (np.trace(covy)-np.trace(S));      taumax = 1e20; taumin = 1e-20;    tau = np.amax([np.amin([tau,taumax]),taumin])

    galpha0 = 1e-20;          balpha0 = 0.1;                 alpha = (2*galpha0 + d)/(tau*np.diag(t(W)*W)+2*galpha0/balpha0)
    gmu0  = 0.00001;            btau0 = 0.1;                   gtau0 = 1e-20;                   SigW = eye(q)
    tauold = 1000

    for epoch in range(epochs):
        Rx = eye(q)+tau*t(W)*W+SigW;              Rxinv = np.linalg.inv(Rx)
        idx = gnomiss;                            n = len(idx)
        dy = mat[idx,:] - np.tile(mu,(n,1));      x = tau * Rxinv * t(W) * t(dy)

        T = t(dy)*t(x);                           trS = np.sum(np.multiply(dy,dy))

        for n in range(len(gmiss)):
            i = gmiss[n]
            dyo = np.copy(mat)[i,nomissidx[i]] - mu[nomissidx[i]]
            Wm = W[missidx[i],:];                                  Wo = W[nomissidx[i],:]
            Rxinv = np.linalg.inv( Rx - tau*t(Wm)*Wm );            ex = tau * t(Wo) * t(m(dyo));                  x = Rxinv * ex
            dym = Wm * x;                                          dy = np.copy(mat)[i,:]

            dy[nomissidx[i]] = dyo;                             dy[missidx[i]] = t(dym)
            yest[i,:] = dy + mu

            T = T + t(m(dy))*t(x);                                    T[missidx[i],:] = T[missidx[i],:] + Wm * Rxinv
            trS = trS + dy*t(m(dy)) +  len(missidx[i])/tau + np.trace( Wm * Rxinv * t(Wm) )

        T = T/N;                trS = trS/N;                        Rxinv = np.linalg.inv(Rx); 
        Dw = Rxinv + tau*t(T)*W*Rxinv + np.diag(alpha)/N;          Dwinv = np.linalg.inv(Dw);
        W = T * Dwinv;

        tau = (d+2*gtau0/N)/(trS-np.trace(t(T)*W)  + (mu*t(m(mu))*gmu0+2*gtau0/btau0)/N)[0,0];
        SigW = Dwinv*(d/N);
        alpha = t((2*galpha0 + d)/ (tau*np.diag(t(W)*W)+np.diag(SigW)+2*galpha0/balpha0))

        if np.abs(np.log10(tau)-np.log10(tauold)) < 1e-6:  break;

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
        for missing_col in missing(original):
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
                from xgboost import XGBRegressor as xgbr
                from xgboost import XGBClassifier as xgbc
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

#-------------------- Data Conversion --------------------#  
def to_cont(x, dummies = True, class_max = 20, return_codes = False, frequency = True, na = "inplace", drop_old = True, drop_other = True,
           na_other_keep = True, columns = None, ascending = True, return_numbers = True):
    
    df = copy(x)
    if class_max == "all": class_max = max(nunique(x))
    if columns is None: change_these = (nunqiue(x[objs(x)])<=class_max).index[nunqiue(x[objs(x)])<=class_max]
    else: 
        if type(columns) == list: change_these = columns
        else: change_these = list(columns)
            
    cleaned = exc(df, change_these)       
    if na_other_keep == True:
            for col in missing(cleaned): df[col+"_nan"] = pd.isnull(df[col])*1
                
    if dummies == True:
        for col in change_these:
            if na == "inplace":
                missed = (isnull(df[col]).index).tolist()
                dummified = pd.get_dummies(df[col], columns = change_these, prefix = col)
                for j in missed:  dummified.iloc[j] = np.nan
            else:  dummified = pd.get_dummies(df[col], columns = change_these, prefix = col, dummy_na = True)
            df = pd.concat([df, dummified],1)
            
        new = exc(df, change_these)
        
        if drop_other == True: new = exc(new, strs(new))
        return new
    else:
        if na != 'inplace': df[change_these] = df[change_these].fillna("Missing_Data")
        if drop_other == True: df = exc(df, strs(cleaned))
        clean = copy(df)
        dicts = {}
        
        for col in change_these:
            if frequency == True:
                counted = tuple((sort(cunique(df[col]), asc = ascending)).index)
                parts = {}; h = 0
                for j in counted: parts[j] = h; h+=1;
            else:
                df[col] = df[col].astype("category")
                parts = dict(enumerate(df[col].cat.categories))
            dicts[col] = parts
        
        if drop_old == True:
            for i in dicts: clean[i] = clean[i].replace(dicts[i])
        else:
            for i in dicts: clean[i+"_codes"] = clean[i].replace(dicts[i])
        
        if return_numbers == True:
            if return_codes == True: return dicts, clean
            else: return clean
        else:
            for i in dicts: 
                dicts[i] = reverse(dicts[i]) 
                if drop_old == True: clean[i] = clean[i].replace(dicts[i])
                else: clean[i+"_codes"] = clean[i].replace(dicts[i])
            if return_codes == True: return dicts, clean
            else: return clean

#-------------
#-------------
#-------------
#-------------
#------------------------------------ DATA VISUALISATIONS ------------------------------------#
#-------------------- Data Frame styles --------------------#
def highlight_larger(s):
    is_max = s > CI(99,s,L=False);    return ['background-color: '+pd_colour if v else '' for v in is_max]
def highlight_smaller(s):
    is_min = s < CI(99,s,U=False);    return ['background-color: '+pd_colour if v else '' for v in is_min]
def nearzero(s):
    is_zero = s < 0.001;              return ['background-color: '+pd_colour if v else '' for v in is_zero]
def false_bad(s):
    is_false = s == False;            return ['background-color: '+pd_colour if v else '' for v in is_false]

#-------------------- Graphical Modules --------------------#
def check_type(x):
    ctd = nunqiue(x);         parts = (((ctd<=15)&(len(x)>15))|((ctd<len(x)*0.01)&(ctd<=20)&(dtypes(x)=='int'))|((dtypes(x)=='str')&(ctd<=15)))
    if dtypes(x) != 'str':
        if parts == True: return 'cat'
        else: return 'cont'
    else: 
        if parts == False: return 'str'
        else: return 'cat'
#-------------    
def plot(x = None, y = None, factor = None, data = None, n = 3, size = 8, smooth = False):
    import matplotlib.pyplot as plt
    import seaborn as sb
    
    cols = []
    if data is not None:
        if x is not None: xlabel = x; cols.append(x)
        if y is not None: ylabel = y; cols.append(y)
        if factor is not None: flabel = factor; cols.append(factor)
        data = data[cols]
    else:
        data = table(x)
        if y is not None and factor is None: 
            data = hcat(data, y)
            data.columns = ["X","Y"]; xlabel = "X"; ylabel = "Y";
        elif y is None and factor is not None:
            data = hcat(data, factor)
            data.columns = ["X","F"]; xlabel = "X"; flabel = "F"
        elif y is None and factor is None:
            xlabel = "X"; data.columns = ["X"]
        else:
            data = hcat(data, y, factor)
            data.columns = ["X","Y","F"]; xlabel = "X"; flabel = "F"; ylabel = "Y"
            
    try: 
        if check_type(data[xlabel]) == 'cat': data[xlabel] = string(data[xlabel])
    except: pass;
    try: 
        if check_type(data[ylabel]) == 'cat': data[ylabel] = string(data[ylabel])
    except: pass;
    try: 
        if check_type(data[flabel]) == 'cat': data[flabel] = string(data[flabel])
    except: pass;

    if y is None and factor is None:
        fig = plt.figure(figsize=(size,size))
        
        if check_type(data[xlabel]) == 'cont':
            fig = sb.kdeplot(data = data[xlabel], linewidth = 3,clip = [min(data[xlabel]),max(data[xlabel])])
            mean_line(data[xlabel])
            plt.ylabel("Frequency"); plt.xlabel(xlabel); plt.title("Kernel Density graph"); plt.show()
        
        elif check_type(data[xlabel]) == 'cat':
            fig = sb.countplot(data[xlabel].fillna("Missing"))
            plt.title("Count graph for "+xlabel); plt.show()
            
    elif y is None:
        if check_type(data[xlabel]) == 'cont': sort_by = xlabel
        else: sort_by = flabel
            
        try:
            df = sort(data, by = sort_by)
            df[sort_by+"_Q"] = qcut(data[sort_by], smooth = smooth, n = n)
            df[sort_by+"_Q"] = string(df[sort_by+"_Q"])
        except: pass;
        fig = plt.figure(figsize=(size,size))
        if check_type(data[xlabel]) == 'cont':
        
            if check_type(data[flabel]) == "cont":
                fig = sb.violinplot(x=xlabel+"_Q", y=flabel, bw='scott' ,scale="width",
                                    cut=min(data[flabel]), inner = None, linewidth =4, data = df)
                plt.setp(fig.get_xticklabels(), rotation=45);    plt.title("Violin graph for "+xlabel+" & "+flabel)
                plt.show()
                
            elif check_type(data[flabel]) == 'cat':
                fig = sb.countplot(x = xlabel+"_Q", hue = flabel, data = df)
                plt.title("Count graph for "+xlabel+" & "+flabel); plt.setp(fig.get_xticklabels(), rotation=45)
                plt.show()
                
        elif check_type(data[xlabel]) == 'cat':
            if check_type(data[flabel]) == "cont":
                fig = sb.countplot(x = xlabel, hue = flabel+"_Q", data = df)
                plt.title("Count graph for "+xlabel+" & "+flabel); plt.setp(fig.get_xticklabels(), rotation=45)
                plt.show()
                
            if check_type(data[flabel]) == "cat":
                fig = sb.countplot(x = xlabel, hue = flabel, data = data)
                plt.title("Count graph for "+xlabel+" & "+flabel); plt.setp(fig.get_xticklabels(), rotation=45)
                plt.show()
                
    elif factor is None:
        if check_type(data[xlabel]) == 'cont':
            
            if check_type(data[ylabel]) == 'cont':
                fig = plt.figure(figsize=(size,size))
                data = notnull(data)
                data[xlabel+"_Q"] = qcut(data[xlabel], n = 30, smooth = True)
                data = (data.groupby(by = xlabel+"_Q").median()+data.groupby(by = xlabel+"_Q").mean())/2
                sb.regplot(x = xlabel, y = ylabel, data = data, ci = None, truncate=True, order=2, color = 'black')
                plt.title("Regression graph for "+xlabel+" & "+ylabel); plt.show()
                
            elif check_type(data[ylabel]) == 'cat':
                fig, (ax1,ax2) = plt.subplots(1,2, sharey = True, figsize = (size*1.5,size))
                sb.boxplot(x = xlabel, y = ylabel, data = data, palette="Set3", linewidth = 3, whis = 1, ax = ax1)
                sb.pointplot(x = xlabel, y = ylabel, data = data, lw=5, ax = ax2, ci = 50, capsize = .1, palette = 'Set1')
                plt.title("Mean PointPlot graph for "+xlabel+" & "+ylabel); plt.show()
        
        elif check_type(data[xlabel]) == 'cat':
            
            if check_type(data[ylabel]) == 'cont':
                fig, (ax1,ax2) = plt.subplots(1,2, sharey = False, figsize = (size*1.5,size))
                sb.boxplot(x = xlabel, y = ylabel, data = data, palette="Set3", linewidth = 3, whis = 1, ax = ax1)
                plt.setp(ax1.get_xticklabels(), rotation=45)
                plt.setp(ax2.get_xticklabels(), rotation=45)
                sb.pointplot(x = xlabel, y = ylabel, data = data, lw=5, ax = ax2, ci = 50, capsize = .1, palette = 'Set1')
                plt.title("Mean PointPlot graph for "+xlabel+" & "+ylabel); plt.show()
                
            elif check_type(data[ylabel]) == 'cat':
                fig = sb.factorplot(x = xlabel, col = ylabel, data = data, size = 5, palette="Set2", col_wrap = 4, kind = "count")
                plt.show()        
    else:
        if check_type(data[flabel]) == 'cont':
            data = notnull(sort(data, by = flabel))
            data[flabel] = string(qcut(data[flabel], smooth = False, n = 4))
        
        elif check_type(data[flabel]) == 'cat':
            data = notnull(data)
            
        if check_type(data[xlabel]) == 'cat':

            if check_type(data[ylabel]) == 'cont':
                try: 
                    fig = plt.figure(figsize=(size,size))
                    fig = sb.barplot(x = xlabel, y = ylabel, hue = flabel, data = data)
                    plt.setp(fig.get_xticklabels(), rotation=45)
                    plt.show()
                except:
                    fig = sb.factorplot(x = xlabel, y = ylabel, data = data, col = flabel, size = 5, capsize=.1, palette="Set2", ci = 70)
                    plt.show()

            elif check_type(data[ylabel]) == 'cat':
                fig = sb.factorplot(x = xlabel, hue = ylabel, data = data, col = flabel, kind = "count", size = 5)
                plt.show()

        elif check_type(data[xlabel]) == 'cont':

            if check_type(data[ylabel]) == 'cont':
                fig = plt.figure(figsize=(size,size))
                fig = sb.lmplot(x = xlabel, y = ylabel, hue = flabel, data = data,robust = True, n_boot = 50, scatter = False, ci = None)
                plt.show()

            elif check_type(data[ylabel]) == 'cat':
                fig = sb.factorplot(x = xlabel, y = ylabel, col = flabel, data = data, palette = "Set3", dodge=True, ci = 70, 
                                    estimator = special_statistic, capsize=.2, n_boot = 100, size = 5)
                plt.show()

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
        df["Column"] = strs(x)[0]
        for col in strs(x)[1:]:
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
    for col in strs(x):
        cols = get(x[col], lower = True, rem = "all", table = True)
        for j in df[df["Column"]==col]["Word"]:
            x["Count="+str(j)] = get(cols[0], count = j)
    return x

#-------------
#-------------
#-------------
#-------------
#------------------------------------ MACHINE LEARNING ------------------------------------#
#-------------------- In Production --------------------#






#------------- Daniel Han-Chen 2017
#------------- https://github.com/danielhanchen/sciblox
#------------- SciBlox v0.01
#-------------