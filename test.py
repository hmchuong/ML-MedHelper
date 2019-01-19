from medhelper.fundus.lrclassifier.classifier import LeftRightClassifier

clf = LeftRightClassifier()
result = clf.predict('/Volumes/Ryan 1TB/data/fundus-caothang/20160527-466105-003.jpg')
print(result)
