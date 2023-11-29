#!/usr/bin/env python
# coding: utf-8

# # Training a Foor Classifier #

# ## Load and Prepare Data ##

# In[3]:


get_ipython().system('pip install -Uqq fastai')
from fastai.vision.all import *


# In[4]:


foodPath = untar_data(URLs.FOOD)


# In[5]:


get_files(foodPath)


# In[6]:


len(get_image_files(foodPath))


# In[13]:


pd.read_json('/Users/danielcrawford/.fastai/data/food-101/train.json')


# In[14]:


pd.read_json('/Users/danielcrawford/.fastai/data/food-101/test.json')


# In[15]:


labelA = 'samosa'
labelB = 'churros'


# In[18]:


#Loop Through all images downloaded
for img in get_image_files(foodPath):
    #Rename Images so that the label (Samosas or Churros) is in the file name
    if labelA in str(img):
        img.rename(f"{img.parent}/{labelA}-{img.name}")
    elif labelB in str(img):
        img.rename(f"{img.parent}/{labelB}-{img.name}")
    else : os.remove(img) #If the images are not part of LabelA or Label B
        
len(get_image_files(foodPath))


# ## Train Model ##

# In[20]:


def GetLabel(fileName):
    return fileName.split('-')[0]

GetLabel("churros-734186.jpg") #testing


# In[21]:


dls = ImageDataLoaders.from_name_func(
    foodPath, get_image_files(foodPath), valid_pct=0.2, seed=420,
    label_func=GetLabel, item_tfms=Resize(224))

dls.train.show_batch()


# In[22]:


learn = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=True)
learn.fine_tune(epochs=10)


# In[24]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(6)


# In[28]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for i in range(0,10):
  #Load random image
  randomIndex = random.randint(0, len(get_image_files(foodPath))-1)
  img = mpimg.imread(get_image_files(foodPath)[randomIndex])
  #Put into Model
  label,_,probs = learn.predict(img)

  #Create Figure using Matplotlib
  fig = plt.figure()
  ax = fig.add_subplot() #Add Subplot (For multiple images)
  imgplot = plt.imshow(img) #Add Image into Plot
  ax.set_title(label) #Set Headline to predicted label

  #Hide numbers on axes
  plt.gca().axes.get_yaxis().set_visible(False)
  plt.gca().axes.get_xaxis().set_visible(False)


# In[29]:


learn.export() #exports model as 'export.pkl' by default


# In[30]:


#fisrt pkl file we can find
modelPath = get_files(foodPath, '.pkl')[0]
modelPath


# In[31]:


learn_inf = load_learner(modelPath)
learn_inf.predict(mpimg.imread(get_image_files(foodPath)[0])) #raw prediction


# In[32]:


learn_inf.dls.vocab #Get the labels


# In[ ]:




