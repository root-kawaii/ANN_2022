import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

Categories = ['Species1','Species2','Species3','Species4','Species5','Species6','Species7','Species8']
flat_data_arr=[]
target_arr=[]
datadir='training_data_final' 
for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(96,96,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
    
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data
y=df.iloc[:,-1] #output data


param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

model.fit(x_train,y_train)

# model.best_params_ contains the best parameters obtained from GridSearchCV

y_pred=model.predict(x_test)

print("The predicted Data is :")

print(y_pred)

print("The actual data is:")

print(np.array(y_test))

print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

url=input('Enter URL of Image :')

img=imread(url)

plt.imshow(img)
plt.show()
img_resize=resize(img,(96,96,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
for ind,val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+Categories[model.predict(l)[0]])