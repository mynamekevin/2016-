#第三题
import pandas as pd
path = 'C:\Users\Administrator\Desktop\yichuan\genotype.dat'
label_path = 'C:\Users\Administrator\Desktop\yichuan\phenotype.txt'
df = pd.read_table(path,delimiter=' ')                      # 读取样本碱基对
label = pd.read_table(label_path,header=None)               # 读取标签值

# H 导入基因数据
gene_path = 'C:\Users\Administrator\Desktop\yichuan\gene_info\\'
gene_file_list = os.listdir('C:\Users\Administrator\Desktop\yichuan\gene_info')
gene_dic={}
import os
i=0
for filer in gene_file_list:
    ff = open(gene_path+filer,'r')							  # 循环打开基因位点文件
    gene_list=[]
    for line in ff.readlines():
        gene_list.append(line.strip('\n'))
    gene_dic[filer] = gene_list 							  # 基因位点数据都存储在字典中
    i=i+1

# print(BasePair35)
datDict={}
datDict35={}
filedir='.\\gene_info\\'
files = os.listdir(filedir)
i=0
for filename in files:
	fullname=os.path.join(filedir,filename)
	# print(fullname)
	condition = pd.read_csv(fullname, header=None)
	# datDict[filename]=list(condition.iloc[:,0])
	gene=list(condition.iloc[:,0])
	comgene=list(set(BasePair35) & set(gene))
	if len(comgene)!=0:
		print(comgene[0])
		# datDict35[filename]=comgene
		while len(comgene)<2:
			comgene.append('0')
		comgene.append(Ka2[Ka2.iloc[:,0]==comgene[0]].iloc[0,1])
		datDict35[filename]=comgene
with open('datDict35.txt', "w") as f:
	f.write(str(datDict35))
datDict35=pd.DataFrame(datDict35)
datDict35.to_csv('datDict35.csv')

import pandas as pd
import os
from collections import Counter
arr0 = pd.read_csv('T30.csv',header=None)
arr1 = pd.read_csv('T31.csv',header=None)
i=0
j=0
# print(arr0[i][j])
tmpSum=0
# sum01avg=(arr0[0][0]+arr1[0][0])/2
# print(pow((arr0[0][0]-sum01avg),2))
while i<3:
	while j<3:
		sum01avg=(arr0[i][j]+arr1[i][j])/2
		tmpSum+=pow((arr0[i][j]-sum01avg),2)/sum01avg+pow((arr1[i][j]-sum01avg),2)/sum01avg
		# print("here is: "+str(i)+" "+str(j))
		j+=1
	i+=1
	j=0
	# print(sum(arr0[col]))
print(tmpSum)
KK=pd.DataFrame(datDict)
Ka2.to_csv('result_3.csv')
print(os.path.exists(".\\gene_info"))
