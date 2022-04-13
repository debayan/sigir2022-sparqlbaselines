import pickle

def split_data(prefix):
    file=open(prefix+'.pickle','rb')
    data=pickle.load(file)
    file.close()
    
    split1,split2,split3,split4,split5=[],[],[],[],[]
    
    split1.append(data[0])
    split1.append(data[1]+data[2]+data[3]+data[4])
    
    split2.append(data[1])
    split2.append(data[0]+data[2]+data[3]+data[4])
    
    split3.append(data[2])
    split3.append(data[1]+data[0]+data[3]+data[4])
    
    split4.append(data[3])
    split4.append(data[1]+data[2]+data[0]+data[4])
    
    split5.append(data[4])
    split5.append(data[1]+data[2]+data[3]+data[0])
    
    file=open(prefix+'1.pickle','wb')
    pickle.dump(split1,file)
    file.close()
    
    file=open(prefix+'2.pickle','wb')
    pickle.dump(split2,file)
    file.close()
    
    file=open(prefix+'3.pickle','wb')
    pickle.dump(split3,file)
    file.close()
    
    file=open(prefix+'4.pickle','wb')
    pickle.dump(split4,file)
    file.close()
    
    file=open(prefix+'5.pickle','wb')
    pickle.dump(split5,file)
    file.close()
    
split_data('split_mix')
