from sklearn.feature_extraction.text import CountVectorizer as CV
import re

vc = CV(analyzer='char_wb', ngram_range=(3,4), min_df=1, token_pattern='[a-z]{2,}')
#fn = vc.fit_transform(name).toarray()
#fn = vc.fit_transform(input3).toarray()
#print vc.get_feature_names()
name = []
input = ['SODR3CS1D3','SODR4_RAVA']
for i in input:
    s = re.findall('(?i)[a-z]{2,}', i)
    name.append(' '.join(s))
vc.fit(name)
print(name)
data1 = vc.transform(name).toarray()
print(data1)
print(vc.get_feature_names())
