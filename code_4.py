# 第四题
import pandas as pd
path = 'C:\Users\Administrator\Desktop\yichuan\genotype.dat'
label_path = 'C:\Users\Administrator\Desktop\yichuan\phenotype.txt'
df = pd.read_table(path,delimiter=' ')                      # 读取样本碱基对
label = pd.read_table(label_path,header=None)               # 读取标签值
# I hash
from itertools import combinations
pheno_path = 'C:\Users\Administrator\Desktop\yichuan\multi_phenos.txt'
import pandas as pd
pheno = pd.read_table(pheno_path,delimiter=' ',header=None)
def group_data(data, degree=3, hash=hash, NAMES=None): 
    new_data = []; combined_names = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        print indicies
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
        if NAMES != None:
            combined_names.append( '+'.join([NAMES[indicies[i]] for i in range(degree)]) )
    if NAMES != None:
        return (np.array(new_data).T, combined_names)
    return np.array(new_data).T
aa=group_data(pheno.values,10)

# G 
pheno_path = 'C:\Users\Administrator\Desktop\yichuan\multi_phenos.txt'
import pandas as pd
pheno = pd.read_table(pheno_path,delimiter=' ',header=None)
dic={}
for ii in range(10):
    y = np.array(pheno.ix[:,ii])
    all_imp = []
    for i in range(50): 
        model_RF = RandomForestClassifier(criterion='entropy',n_estimators=160, max_depth=160, min_samples_split=7, min_samples_leaf=10)
        model_RF.fit(X,y)
        important = model_RF.feature_importances_
        imp_index = np.argsort(-important)[:50] 
        all_imp.extend(imp_index)
    all_impp = pd.Series(all_imp)
    value_count = all_impp.value_counts()
    value_count_index = value_count[0:50].index
    dic[ii] = df.columns[value_count_index]
    print ii