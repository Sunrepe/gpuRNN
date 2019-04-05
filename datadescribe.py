'''
描述所有数据
'''
import numpy as np
import os
import pandas as pd

def Read__mean_2(filename):
    '''
    获得所有且分点信息，同时将所有数据进行（绝对值、去噪操作）
    :param filename:
    :return: 转置的8*N 的预处理的原始数据
    '''
    # f_csv = csv.reader(filename)
    my_matrix = np.loadtxt(filename, dtype='int', delimiter=",")
    return my_matrix


def get_lei(sq):
    alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    return alllei.index(sq)


def get_label(ch, num_classes=10):
    return np.eye(num_classes)[ch]


# 数据标准化方案1
def z_score(data_x):
    x_m = np.mean(data_x)
    x_p = np.std(data_x)
    x = (data_x - x_m) / x_p
    return x


def getAllPeople(foldname):
    '''
        根据文件夹获得所有需要测试的人,主要是根据文件夹获得文件夹下所有人名
    '''
    _person = set()
    for filename in os.listdir(foldname):
        oa, ob, oc = filename.split('_')
        _person.add(oa)
    return list(_person)


def main():
    foldname = './data/test3/'
    personlist = getAllPeople(foldname)
    oblist = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no', 'finger', 'snap']
    ave_k = [7.71,11.51,12.01,9.87,11.59,11.47,10.36,10.85,10.97,11.67]
    df_savepath1 = './dataAVE.csv'
    df_savepath2 = './dataLONG.csv'
    df_savepath3 = './dataNUM.csv'
    peop_info1 = []  # 保存AVE
    peop_info2 = []  # 保存LONG
    peop_info3 = []  # 保存NUM
    peop_info4 = []
    num_all_pose = 0
    for person in personlist:
        print(person)
        pose_info1 = []
        pose_info2 = []
        pose_info3 = []
        pose_info4 = []
        for ob in oblist:
            filenames = foldname + person+'_'+ob+'_b.txt'
            if os.path.exists(filenames):
                data = Read__mean_2(filenames)
                cutting = Read__mean_2(foldname + person + '_' + ob + '_c.txt')
                num_pose = 0
                ave_pose = []
                long_pose = []
                for cut in range(0, len(cutting)):
                    if cut == 0:
                        tmp_data = data[0:cutting[cut], :]
                    else:
                        tmp_data = data[cutting[cut - 1]:cutting[cut], :]
                    _len = tmp_data.shape[0]
                    if _len > 800:
                        pass
                    else:
                        num_pose += 1
                        num_all_pose += 1
                        ave_pose.append(np.mean(abs(tmp_data)))
                        long_pose.append(_len)
                pose_info1.append(sum(ave_pose)/num_pose)
                pose_info2.append(sum(long_pose)/num_pose)
                pose_info3.append(num_pose)
            else:
                pose_info1.append(0.0)
                pose_info2.append(0.0)
                pose_info3.append(0.0)
        peop_info1.append(pose_info1)
        peop_info2.append(pose_info2)
        peop_info3.append(pose_info3)
        peop_info4.append(np.array(pose_info1)-np.array(ave_k))
    df1 = pd.DataFrame(peop_info1, index=personlist, columns=oblist)
    df2 = pd.DataFrame(peop_info2, index=personlist, columns=oblist)
    df3 = pd.DataFrame(peop_info3, index=personlist, columns=oblist)
    df4 = pd.DataFrame(peop_info4, index=personlist, columns=oblist)
    df1.to_csv(df_savepath1)
    df2.to_csv(df_savepath2)
    df3.to_csv(df_savepath3)
    df4.to_csv('./dataEMGmins')
    print("All test pose:", num_all_pose)


if __name__ == '__main__':
    main()
