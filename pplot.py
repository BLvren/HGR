import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt



def getmean(filename, chunksize, index,epoch):
    mean_list=[]
    i=0
    for chunk in pd.read_csv(filename, chunksize=chunksize,skip_blank_lines=True):

        temp=chunk.iloc[:, index].mean()
        i =i +1
        if(i==epoch): cov_chunk=chunk
        mean_list.append(temp)
    return  mean_list, cov_chunk

# Hscore resnet50 seg&original train accuracy 对比

# mean_orig, cov_chunk1=getmean('original_batch_CE_train_acc.csv',37,1,50)
# mean_seg, cov_chunk2=getmean('seg_batch_CE_train_acc.csv',37,1,50)
# epoch1=range(1,len(mean_orig)+1)
# epoch2=range(1,len(mean_seg)+1)
# plt.plot(epoch1, mean_orig,'r')
# plt.plot(epoch2,mean_seg,'g')
# plt.legend(['with-bg','without-bg'])
# plt.title('CE-resnet18-pre train accuracy')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()

# batch=range(1,38)
# plt.plot(batch, cov_chunk1.iloc[:,0],'r',batch,cov_chunk2.iloc[:,0],'g')
# plt.legend(['with-bg','without-bg'])
# plt.title('CE-resnet18-pre train accuracy on epoch-50')
# plt.xlabel('epoch-50')
# plt.ylabel('acc')
# plt.show()
#
# # Hscore resnet50 seg&original test accuracy 对比
#
# mean_orig, cov_chunk1=getmean('original_batch_CE_test_acc.csv',10,1,50)
# mean_seg, cov_chunk2=getmean('seg_batch_CE_test_acc.csv',10,1,50)
# epoch1=range(1,len(mean_orig)+1)
# epoch2=range(1,len(mean_seg)+1)
# plt.plot(epoch1, mean_orig,'r')
# plt.plot(epoch2,mean_seg,'g')
# plt.legend(['with-bg','without-bg'])
# plt.title('CE-resnet18pre test accuracy')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()
#
# batch=range(1,11)
# plt.plot(batch, cov_chunk1.iloc[:,0],'r',batch,cov_chunk2.iloc[:,0],'g')
# plt.legend(['with-bg','without-bg'])
# plt.title('CE-resnet18-pre test accuracy on epoch-50')
# plt.xlabel('epoch-50')
# plt.ylabel('acc')
# plt.show()

# Hscore resnet50 seg&original train Hscore 对比

mean_orig, cov_chunk1=getmean('original_trainloss.csv',37,1,50)
mean_seg, cov_chunk2=getmean('seg_trainloss.csv',37,1,50)
epoch1=range(1,len(mean_orig)+1)
epoch2=range(1,len(mean_seg)+1)
plt.plot(epoch1, mean_orig,'r')
plt.plot(epoch2,mean_seg,'g')
plt.legend(['with-bg','without-bg'])
plt.title('Hscore-resnet50-pre train Hscore')
plt.xlabel('epoch')
plt.ylabel('Hscore')
plt.show()

batch=range(1,38)
plt.plot(batch, cov_chunk1.iloc[:,1],'r',batch,cov_chunk2.iloc[:,1],'g')
plt.legend(['with-bg','without-bg'])
plt.title('Hscore-resnet50-pre train Hscore on epoch-50')
plt.xlabel('epoch-50')
plt.ylabel('Hscore')
plt.show()










# batch=range(1,38)
# plt.plot(batch, cov_chunk1.iloc[:,1],'r',batch,cov_chunk2.iloc[:,1],'g')
# plt.legend(['with-bg','without-bg'])
# plt.xlabel('epoch-60')
# plt.ylabel('Hscore')
# plt.show()

# original_acc='original_trainacc.csv'
# seg_acc='seg_trainacc.csv'
#
# mean_orig, cov_chunk1=getmean(original_acc)
# mean_seg, cov_chunk2=getmean(seg_acc)
# epoch1=range(1,107)
# epoch2=range(1,101)
# plt.plot(epoch1, mean_orig,'r')
# plt.plot(epoch2,mean_seg,'g')
# plt.legend(['with-bg','without-bg'])
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()

# original_acc='original_testacc.csv'
# seg_acc='seg_testacc.csv'
#
# mean_orig, cov_chunk1=getmean(original_acc)
# mean_seg, cov_chunk2=getmean(seg_acc)
# epoch1=range(1,103)
# epoch2=range(1,111)
# plt.plot(epoch1, mean_orig,'r')
# plt.plot(epoch2,mean_seg,'g')
# plt.legend(['with-bg','without-bg'])
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()