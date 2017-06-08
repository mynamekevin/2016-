# 问题一
# A 导入数据
import pandas as pd
path = 'C:\Users\Administrator\Desktop\yichuan\genotype.dat'
label_path = 'C:\Users\Administrator\Desktop\yichuan\phenotype.txt'
df = pd.read_table(path,delimiter=' ')						# 读取样本碱基对
label = pd.read_table(label_path,header=None)               # 读取标签值
df.head(7)

# B 数值编码
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_coder = df.apply(lambda x:le.fit_transform(x))           # apply对每一列编码
df_coder.head(7)
df_coder.to_csv('result_1.csv')

#问题二
# C 卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2                  # 导入卡方检验函数
#选择K个最好的特征，返回选择特征后的数据
re=SelectKBest(chi2, k=1000).fit_transform(df_coder, label) # 选择1000个最好的特征

# D 随机森林
all_imp = []
for i in range(300): # 迭代300次
    model_RF = RandomForestClassifier(criterion='entropy',n_estimators=160, max_depth=160, 
    	min_samples_split=7, min_samples_leaf=10)
    model_RF.fit(df_coder,label)
    important = model_RF.feature_importances_
    imp_index = np.argsort(-important)[:100]                 # 筛选前100个最重要的特征
    all_imp.extend(imp_index)

# E 统计300次迭代后的频率
all_impp = pd.Series(all_imp)
value_count = all_impp.value_counts()                         # 统计频率
value_count_index = value_count[0:100].index                  # 频率最高的前100个索引

# F 随机森林寻参
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
pipeline = Pipeline([('clf', RandomForestClassifier(criterion='entropy'))])
parameters = {
    'clf__n_estimators': (150,160,170),            			  # 设置寻参区间，根据结果不断缩小区间
    'clf__max_depth': (150,160,170),
    'clf__min_samples_split': (6，7，8),
    'clf__min_samples_leaf': (9，10，11)
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(de_coder,label[0].values)
print('最佳效果：%0.3f' % grid_search.best_score_)    		  # 设置评价标准为f1
print('最优参数：')
best_parameters = grid_search.best_estimator_.get_params()    # 返回最优参数值
for param_name in sorted(parameters.keys()):
    print('\t%s: %r' % (param_name, best_parameters[param_name]))

# G 检验（划分+随机森林预测）
X=pd.DataFrame(re).as_matrix()
y = np.array(label).T[0]
y_prob = y.copy() 
from sklearn.cross_validation import KFold                    # 导入kfold函数
kf = KFold(len(y), n_folds=10,shuffle=True)					  # 设置10折交叉验证
test_RF = RandomForestClassifier(criterion='entropy',n_estimators=160, max_depth=100, 
	min_samples_split=7, min_samples_leaf=9)
for train_index, test_index in kf: 							  # 划分训练集和测试集索引
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    test_RF.fit(X_train,y_train) 							  # 训练
    y_prob[test_index] = test_RF.predict(X_test) 			  # 预测
print np.mean(y==y_prob)									  # 求准确率

